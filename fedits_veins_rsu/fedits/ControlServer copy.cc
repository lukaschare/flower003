#include "fedits/ControlServer.h"

#include <arpa/inet.h>
#include <cmath>
#include <cstring>
#include "nlohmann/json.hpp"

using namespace omnetpp;

namespace fedits {

Define_Module(ControlServer);

ControlServer::~ControlServer() {
    closeConn_();
    closeListen_();
    cancelAndDelete(wait_msg_);
    cancelAndDelete(tick_msg_);
}

void ControlServer::initialize() {
    port_ = par("port").intValue();
    rsu_x_m_ = par("rsuX").doubleValue();
    rsu_y_m_ = par("rsuY").doubleValue();
    rsu_r_m_ = par("rsuR").doubleValue();
    step_s_ = par("stepS").doubleValue();

    max_goodput_mbps_ = par("maxGoodputMbps").doubleValue();
    min_goodput_mbps_ = par("minGoodputMbps").doubleValue();
    alpha_ = par("alpha").doubleValue();
    base_rtt_ms_ = par("baseRttMs").doubleValue();

    setupListen_();

    wait_msg_ = new cMessage("WAIT_CMD");
    tick_msg_ = new cMessage("TICK");

    // WAIT 在同一时刻优先执行，保证 barrier 卡住仿真
    wait_msg_->setSchedulingPriority(-100);
    // TICK 尽量在 TraCI 更新之后执行
    tick_msg_->setSchedulingPriority(10);

    mode_ = Mode::WAIT_CMD;
    scheduleAt(simTime(), wait_msg_);
}

void ControlServer::handleMessage(cMessage* msg) {
    if (msg == wait_msg_) {
        mode_ = Mode::WAIT_CMD;

        // barrier: 阻塞等待 RPC（训练期间 SUMO/Veins 就停在这里）
        std::string line = recvLineBlocking_();
        if (line.empty()) {
            scheduleAt(simTime(), wait_msg_);
            return;
        }

        json req;
        try { req = json::parse(line); }
        catch (...) {
            sendJsonAndClose_({{"ok", false}, {"error", "json_parse_failed"}});
            scheduleAt(simTime(), wait_msg_);
            return;
        }

        std::string cmd = req.value("cmd", "");
        if (cmd == "get_state") {
            json resp = handleGetState_(req);
            sendJsonAndClose_(resp);
            scheduleAt(simTime(), wait_msg_);
            return;
        }

        if (cmd == "simulate_downlink") {
            pending_cmd_ = cmd;
            pending_req_ = req;
            startDownlink_(req);
            mode_ = Mode::RUN_XFER;
            scheduleAt(simTime(), tick_msg_);
            return;
        }

        if (cmd == "simulate_uplink") {
            pending_cmd_ = cmd;
            pending_req_ = req;
            startUplink_(req);
            mode_ = Mode::RUN_XFER;
            scheduleAt(simTime(), tick_msg_);
            return;
        }

        sendJsonAndClose_({{"ok", false}, {"error", "unknown_cmd"}, {"cmd", cmd}});
        scheduleAt(simTime(), wait_msg_);
        return;
    }

    if (msg == tick_msg_) {
        if (mode_ != Mode::RUN_XFER) return;

        tickOnce_();

        // 必须推进到 deadline（round end），然后回到 barrier
        if (simTime().dbl() + 1e-9 >= deadline_) {
            json resp = buildXferResp_();
            sendJsonAndClose_(resp);

            sessions_.clear();
            pending_cmd_.clear();
            pending_req_.clear();

            mode_ = Mode::WAIT_CMD;
            scheduleAt(simTime(), wait_msg_);
            return;
        }

        scheduleAt(simTime() + SimTime(step_s_), tick_msg_);
        return;
    }
}

// ---------------- socket ----------------

void ControlServer::setupListen_() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw cRuntimeError("socket() failed");

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0)
        throw cRuntimeError("bind() failed on port %d", port_);
    if (::listen(listen_fd_, 16) < 0)
        throw cRuntimeError("listen() failed");

    EV_INFO << "[fedits] ControlServer listen 0.0.0.0:" << port_ << "\n";
}

void ControlServer::closeConn_() {
    if (conn_fd_ >= 0) {
        ::close(conn_fd_);
        conn_fd_ = -1;
    }
    inbuf_.clear();
}

void ControlServer::closeListen_() {
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

std::string ControlServer::recvLineBlocking_() {
    sockaddr_in cli{};
    socklen_t len = sizeof(cli);
    conn_fd_ = ::accept(listen_fd_, (sockaddr*)&cli, &len);
    if (conn_fd_ < 0) return "";

    inbuf_.clear();
    char buf[4096];

    while (true) {
        ssize_t n = ::recv(conn_fd_, buf, sizeof(buf), 0);
        if (n <= 0) break;
        inbuf_.append(buf, buf + n);

        auto pos = inbuf_.find('\n');
        if (pos != std::string::npos) {
            return inbuf_.substr(0, pos);
        }
        if (inbuf_.size() > 1'000'000) break;
    }
    // allow EOF-terminated JSON too
    return inbuf_;
}

void ControlServer::sendJsonAndClose_(const json& j) {
    if (conn_fd_ < 0) return;
    std::string out = j.dump();
    out.push_back('\n');
    ::send(conn_fd_, out.data(), out.size(), 0);
    closeConn_();
}

// ---------------- traci helpers ----------------

veins::TraCIScenarioManager* ControlServer::traciMgr_() const {
    return veins::TraCIScenarioManagerAccess().get();
}

veins::TraCIMobility* ControlServer::mobilityOf_(cModule* host) const {
    if (!host) return nullptr;
    cModule* m = host->getSubmodule("veinsmobility");
    if (!m) m = host->getSubmodule("mobility");
    if (!m) return nullptr;
    return dynamic_cast<veins::TraCIMobility*>(m);
}

// ---------------- link helpers ----------------

double ControlServer::distToRsu_(const veins::Coord& p) const {
    double dx = p.x - rsu_x_m_;
    double dy = p.y - rsu_y_m_;
    return std::sqrt(dx*dx + dy*dy);
}

bool ControlServer::inRange_(const veins::Coord& p) const {
    return distToRsu_(p) <= rsu_r_m_;
}

double ControlServer::goodputMbps_(const veins::Coord& p) const {
    double d = distToRsu_(p);
    if (d >= rsu_r_m_) return 0.0;
    double x = d / rsu_r_m_;
    double g = max_goodput_mbps_ * (1.0 - std::pow(x, alpha_));
    if (g < min_goodput_mbps_) g = min_goodput_mbps_;
    return g;
}

double ControlServer::rttMs_(const veins::Coord& p) const {
    double d = distToRsu_(p);
    return base_rtt_ms_ + (d / 3e8) * 2.0 * 1000.0;
}

// ---------------- rpc handlers ----------------

json ControlServer::handleGetState_(const json& req) {
    (void)req;

    auto* mgr = traciMgr_();
    json vehicles = json::object();

    const auto& hosts = mgr->getManagedHosts();  // sumoId -> cModule*
    for (const auto& kv : hosts) {
        const std::string& sumoId = kv.first;
        cModule* host = kv.second;
        auto* mob = mobilityOf_(host);
        if (!mob) continue;

        veins::Coord pos = mob->getPositionAt(simTime());
        vehicles[sumoId] = {
            {"x_m", pos.x},
            {"y_m", pos.y},
            {"in_range", inRange_(pos)}
        };
    }

    return {{"ok", true}, {"t_sim", simTime().dbl()}, {"vehicles", vehicles}};
}

void ControlServer::startDownlink_(const json& req) {
    sessions_.clear();
    deadline_ = req.value("deadline", simTime().dbl());

    long long size_bytes = req.value("size_bytes", 0LL);
    std::vector<std::string> veh_ids = req.value("veh_ids", std::vector<std::string>{});

    double t0 = simTime().dbl();
    for (const auto& vid : veh_ids) {
        Session s;
        s.veh_id = vid;
        s.dir = Dir::DL;
        s.start_time = t0;
        s.bytes_total = size_bytes;
        s.bytes_left = size_bytes;
        sessions_.push_back(s);
    }
}

void ControlServer::startUplink_(const json& req) {
    sessions_.clear();
    deadline_ = req.value("deadline", simTime().dbl());

    long long size_bytes = req.value("size_bytes", 0LL);
    std::vector<std::string> veh_ids = req.value("veh_ids", std::vector<std::string>{});
    json start_times = req.value("start_times", json::object());

    for (const auto& vid : veh_ids) {
        if (!start_times.contains(vid)) continue;

        double st = start_times[vid].get<double>();
        Session s;
        s.veh_id = vid;
        s.dir = Dir::UL;
        s.start_time = st;
        s.bytes_total = size_bytes;
        s.bytes_left = size_bytes;
        sessions_.push_back(s);
    }
}

// ---------------- tick loop ----------------

bool ControlServer::allFinished_() const {
    for (const auto& s : sessions_) if (!s.finished) return false;
    return true;
}

void ControlServer::tickOnce_() {
    double now = simTime().dbl();

    auto* mgr = traciMgr_();
    const auto& hosts = mgr->getManagedHosts();

    for (auto& s : sessions_) {
        if (s.finished) continue;

        if (now + 1e-9 < s.start_time) {
            continue; // per-veh start time not reached yet
        }
        s.started = true;

        if (now > deadline_ + 1e-9) {
            s.finished = true;
            s.res.ok = false;
            s.res.reason = "deadline";
            continue;
        }

        auto it = hosts.find(s.veh_id);
        if (it == hosts.end()) {
            s.finished = true;
            s.res.ok = false;
            s.res.reason = "veh_missing";
            continue;
        }

        auto* mob = mobilityOf_(it->second);
        if (!mob) {
            s.finished = true;
            s.res.ok = false;
            s.res.reason = "mobility_missing";
            continue;
        }

        veins::Coord pos = mob->getPositionAt(simTime());
        if (!inRange_(pos)) {
            s.finished = true;
            s.res.ok = false;
            s.res.reason = "left_coverage";
            continue;
        }

        double g_mbps = goodputMbps_(pos);
        double rtt_ms = rttMs_(pos);
        s.res.goodput_mbps = g_mbps;
        s.res.rtt_ms = rtt_ms;

        double rate_Bps = (g_mbps * 1e6) / 8.0;
        long long deliver = (long long)std::floor(rate_Bps * step_s_);
        if (deliver <= 0) deliver = 1;

        s.bytes_left -= deliver;

        if (s.bytes_left <= 0) {
            // refine completion time within this step
            double delivered_to_finish = (double)(deliver + s.bytes_left);
            double step_bytes = std::max(1.0, rate_Bps * step_s_);
            double frac = delivered_to_finish / step_bytes;
            double t_done = now + frac * step_s_;

            s.finished = true;
            s.res.ok = true;
            s.res.t_done = t_done;
            s.res.reason.clear();
        }
    }
}

json ControlServer::buildXferResp_() const {
    json results = json::object();
    for (const auto& s : sessions_) {
        results[s.veh_id] = {
            {"ok", s.res.ok},
            {"t_done", s.res.ok ? s.res.t_done : -1.0},
            {"goodput_mbps", s.res.goodput_mbps},
            {"rtt_ms", s.res.rtt_ms},
            {"reason", s.res.ok ? "" : s.res.reason},
        };
    }

    return {
        {"ok", true},
        {"cmd", pending_cmd_},
        {"t_sim", simTime().dbl()},
        {"deadline", deadline_},
        {"results", results},
    };
}

} // namespace fedits

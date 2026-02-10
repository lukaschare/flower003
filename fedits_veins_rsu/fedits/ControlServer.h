#pragma once

#include <omnetpp.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

// #include "veins/modules/mobility/traci/TraCIScenarioManagerAccess.h"
#include "veins/modules/mobility/traci/TraCIScenarioManager.h"
#include "veins/base/utils/FindModule.h"

#include "veins/modules/mobility/traci/TraCIMobility.h"
#include "veins/base/utils/Coord.h"

namespace fedits {

using json = nlohmann::json;

struct XferRes {
    bool ok = false;
    double t_done = -1.0;
    double goodput_mbps = 0.0;
    double rtt_ms = 0.0;
    std::string reason;
};

enum class Dir { DL, UL };

struct Session {
    std::string veh_id;
    Dir dir;

    double start_time = 0.0;
    long long bytes_total = 0;
    long long bytes_left = 0;

    bool started = false;
    bool finished = false;

    XferRes res;
};

class ControlServer : public omnetpp::cSimpleModule {
  public:
    ControlServer() = default;
    ~ControlServer() override;

  protected:
    void initialize() override;
    void handleMessage(omnetpp::cMessage* msg) override;

  private:
    // ----- params (from ini) -----
    int port_ = 9999;

    double rsu_x_m_ = 500.0;
    double rsu_y_m_ = 500.0;
    double rsu_r_m_ = 300.0;

    double step_s_ = 0.1;

    // MVP link model
    double max_goodput_mbps_ = 12.0;
    double min_goodput_mbps_ = 0.5;
    double alpha_ = 2.0;
    double base_rtt_ms_ = 10.0;

    // ----- socket -----
    int listen_fd_ = -1;
    int conn_fd_ = -1;
    std::string inbuf_;

    // ----- barrier + run state -----
    enum class Mode { WAIT_CMD, RUN_XFER } mode_ = Mode::WAIT_CMD;
    omnetpp::cMessage* wait_msg_ = nullptr;
    omnetpp::cMessage* tick_msg_ = nullptr;

    double deadline_ = 0.0;
    std::vector<Session> sessions_;

    std::string pending_cmd_;
    json pending_req_;

  private:
    // socket helpers
    void setupListen_();
    void closeConn_();
    void closeListen_();
    std::string recvLineBlocking_();
    void sendJsonAndClose_(const json& j);

    // traci helpers
    veins::TraCIScenarioManager* traciMgr_() const;
    veins::TraCIMobility* mobilityOf_(omnetpp::cModule* host) const;

    // link helpers
    double distToRsu_(const veins::Coord& p) const;
    bool inRange_(const veins::Coord& p) const;
    double goodputMbps_(const veins::Coord& p) const;
    double rttMs_(const veins::Coord& p) const;

    // rpc handlers
    json handleGetState_(const json& req);
    void startDownlink_(const json& req);
    void startUplink_(const json& req);

    // tick loop
    void tickOnce_();
    bool allFinished_() const;
    json buildXferResp_() const;
};

} // namespace fedits

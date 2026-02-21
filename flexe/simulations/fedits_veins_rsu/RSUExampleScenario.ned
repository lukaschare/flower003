//
// Copyright (C) 2017 Christoph Sommer <sommer@ccs-labs.org>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//


// network RSUExampleScenario extends Scenario
//{
//    submodules:
//        rsu[1]: RSU {
//            @display("p=150,140;i=veins/sign/yellowdiamond;is=vs");
//        }
// }


// package org.car2x.veins.examples.fedits_veins_rsu;

// package org.car2x.veins.examples.rsu;

//之前的
//import org.car2x.veins.nodes.RSU;
//import org.car2x.veins.nodes.Scenario;

//import fedits.ControlServer;

//network RSUExampleScenario extends Scenario
//{
//    submodules:
//        rsu[1]: RSU {
//            @display("p=150,140;i=veins/sign/yellowdiamond;is=vs");
//        }
//        control: ControlServer {
//            @display("p=80,80");
//        }
//}

//package fedits_veins_rsu;

import veins.base.connectionManager.ConnectionManager;
import veins.modules.mobility.traci.TraCIScenarioManagerLaunchd;
import flexe.application.*;          // ← FLEXE 的真实发包模块
import veins.modules.application.ieee80211p.*;  // 如果需要标准 WAVE

network FeditsRsuNetwork
{
    parameters:
        int numVehicles = default(50);   // 你可以改大
        string sumoConfig = default("erlangen.sumo.cfg");

    submodules:
        connectionManager: ConnectionManager;
        manager: TraCIScenarioManagerLaunchd {
            parameters:
                @display("p=100,100");
                launchConfig = xmldoc("erlangen.launchd.xml");
                // ... 其他你的 SUMO 参数
        }
        rsu[1]: FlexeRSU {   // FLEXE 通常提供 RSU 模块，或用 SimpleRSU11p 改
            @display("p=500,500");
        }
        vehicle[numVehicles]: FlexeNode {   // 关键！真实发包 + gRPC 的车辆
            @display("i=block/wifi");
        }
}
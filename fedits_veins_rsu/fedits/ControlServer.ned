package fedits;

simple ControlServer
{
    parameters:
        // 强制指定 C++ 类名为 fedits::ControlServer
        @class("fedits::ControlServer");
        
        int port = default(9999);

        double rsuX = default(500);
        double rsuY = default(500);
        double rsuR = default(300);

        double stepS = default(0.1);

        double maxGoodputMbps = default(12);
        double minGoodputMbps = default(0.5);
        double alpha = default(2.0);
        double baseRttMs = default(10.0);
}

import time
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass

# ==========================================
# 1. 模拟配置类 (Mock Config)
#    完全模仿 OrchestratorConfig 的结构
# ==========================================
@dataclass
class MockConfig:
    ci_mode: str = "electricitymaps"
    # 【重要】在这里填入你的真实 Token 用于测试
    # emaps_token: str = "YOUR_REAL_TOKEN_HERE" 
    emaps_token: str = "eyEFSfkLXrarj0rSMNRn" 
    
    
    # 区域代码：西班牙是 ES，加泰罗尼亚是 ES-CT
    emaps_zone: str = "ES"  
    
    ci_cache_s: int = 10      # 测试时缓存设短一点
    ci_timeout_s: float = 5.0 # 超时时间
    ci_g_per_kwh: float = 153.0 # API 挂了时的保底值

# ==========================================
# 2. 你的 Provider 类 (完全复制过来的逻辑)
# ==========================================
class CarbonIntensityProvider:
    def __init__(self, cfg) -> None:
        self.mode = (cfg.ci_mode or "fixed").lower().strip()
        self.fixed_ci = float(cfg.ci_g_per_kwh)
        self.token = (cfg.emaps_token or "").strip()
        self.zone = (cfg.emaps_zone or "ES").strip()
        self.base_url = "https://api.electricitymap.org"
        self.cache_s = int(cfg.ci_cache_s) if cfg.ci_cache_s else 300
        self.timeout_s = float(cfg.ci_timeout_s) if cfg.ci_timeout_s else 5.0
        
        self._cache_ts: float = 0.0
        self._cache_ci: float = self.fixed_ci

    def get_ci_g_per_kwh(self) -> float:
        # 如果不是动态模式或没 Token，直接返回固定值
        if self.mode != "electricitymaps" or not self.token:
            print(f"[DEBUG] Mode is {self.mode} or no token. Using Fixed.")
            return self.fixed_ci

        now = time.time()
        # 检查缓存
        if self._cache_ts > 0 and (now - self._cache_ts) < self.cache_s:
            print(f"[DEBUG] Using Cache (Age: {now - self._cache_ts:.1f}s)")
            return float(self._cache_ci)

        # 构造请求
        params = {"zone": self.zone}
        url = f"{self.base_url}/v3/carbon-intensity/latest?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"auth-token": self.token, "accept": "application/json"}
        )

        try:
            print(f"[Network] Requesting {url} ...")
            start_t = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                print(f"[Network] Response Code: {resp.status}")
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                
                raw_data = resp.read().decode("utf-8")
                # print(f"[DEBUG] Raw Body: {raw_data}") # 如果想看完整返回就把这行解注
                data = json.loads(raw_data)
            
            duration = time.time() - start_t
            
            # 提取数据
            ci = float(data.get("carbonIntensity", self.fixed_ci))
            print(f"[Success] CI={ci} g/kWh (Time taken: {duration:.2f}s)")
            
            # 更新缓存
            self._cache_ts = now
            self._cache_ci = ci
            return ci
            
        except Exception as e:
            print(f"[Error] API failed: {e}")
            print(f"[Fallback] Using fixed value: {self.fixed_ci}")
            # 失败时也更新缓存时间，防止重试风暴
            self._cache_ts = now
            self._cache_ci = self.fixed_ci
            return self.fixed_ci

# ==========================================
# 3. 执行测试
# ==========================================
if __name__ == "__main__":
    print("=== Starting API Test ===\n")
    
    # 实例化配置
    cfg = MockConfig()
    
    if cfg.emaps_token == "YOUR_REAL_TOKEN_HERE":
        print("❌ 错误：请先在代码里把 YOUR_REAL_TOKEN_HERE 替换成你的真实 Token！")
        exit(1)

    # 实例化 Provider
    provider = CarbonIntensityProvider(cfg)

    # 第一次调用：应该走网络请求
    print(">>> Call #1 (Should hit API)")
    val1 = provider.get_ci_g_per_kwh()
    print(f"Result #1: {val1}\n")

    # 第二次调用：应该走缓存 (模拟同一轮次内多次调用)
    print(">>> Call #2 (Should hit Cache)")
    val2 = provider.get_ci_g_per_kwh()
    print(f"Result #2: {val2}\n")

    print("=== Test Finished ===")
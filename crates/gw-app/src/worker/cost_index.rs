//! 最优控制调度器(代价指数)的纯计算部分。调度器在 `entries` 锁内收集候选快照,
//! 交给这里算指数、做决策;本模块不碰任何锁与运行态,便于单测。
//!
//! ## 模型(2026-09-30 调度研究报告,caio-sched-sim/report)
//!
//! 上游 prompt 缓存按账号隔离,换号 = 冷启动,多花 `x·Δr` 积分(x = 输入 token)。
//! 每次选号是一个最优停止问题:在热号上等,还是立刻去冷号?一步前瞻的代价指数:
//!
//! ```text
//! I_a = 单价_a·(b0 + x·r(热_a)/1e5) + κ·E[W_a] + P429_a·(κ·t_retry + η)
//! ```
//!
//! 取最小。κ = 多等 1 秒愿意付的积分;η = 每次 429 的风险成本(封号风险代理)。
//! 推论:热号上值得等的上限 `W* = 单价·x·Δr/κ` 与上下文大小成正比 —— 旧版对所有请求
//! 一律等 1.5s,对 40 万上下文等得太短、对 3 万上下文等得太长。
//! 常数来自生产计费标定:热 0.342、冷 1.197 积分/10 万输入 token,基础 0.086 积分/请求。
//!
//! ## 2026-10-03 修正(上线 3 天实测发现的两处逻辑缺陷)
//!
//! 1. **热不是 0/1**。原先「TTL 内 = 完全热,过了 = 和陌生号一样冷」,于是线程一过 5 分钟
//!    就被当成新线程,平局规则(在途少 → 最久未选)反而**避开**它刚用过的号。实测同号续轮
//!    过了 5 分钟仍有约 33% 命中(陌生号约 20%),回原号是白赚。改为按「距上次在该号起跑」
//!    查实测曲线得到热度 `w ∈ [0,1]`(回到该号能省下的冷热差价比例),并把记忆延长到 6 个 TTL。
//! 2. **冷启动不只是贵,还慢**。冷预填首包每 10 万输入多约 1.2 秒,原公式没算,20 万上下文
//!    只肯等 1.7 秒 —— 换号后照样要多等 2–3 秒。现在冷的那部分 `(1-w)` 同时计积分差与时间差:
//!
//! ```text
//! I_a = 单价_a·(b0 + x·(w·r热 + (1-w)·r冷)/1e5) + κ·(E[W_a] + (1-w)·c预填·x/1e5) + P429_a·(κ·t_retry + η)
//! ```
//!
//! 另外,代价打平时(多个陌生号)先挑**近期挂着的活跃线程少**的号,而不是只看此刻在途数:
//! 在途数会把「会话恰好在两轮之间」的号误判成空闲,新线程挤上去,原线程回来就撞车溢出。

use std::collections::HashMap;
use std::time::{Duration, Instant};

use gw_core::config::SchedulerConfig;

/// 每请求基础积分(标定值)。
pub(crate) const CREDIT_BASE: f64 = 0.086;
/// 缓存命中时每 10 万输入 token 的积分(标定值)。
pub(crate) const RATE_WARM: f64 = 0.342;
/// 冷启动时每 10 万输入 token 的积分(标定值)。冷热差 Δr = 0.855。
pub(crate) const RATE_COLD: f64 = 1.197;
/// 冷预填比命中缓存多出的首包秒数,每 10 万输入 token(标定值)。2026-10-03 低负载实测
/// (opus 系,同输入量分档的冷/热首包 p50 之差):5–30 万输入每 10 万约 1.1–1.5 秒。
pub(crate) const COLD_PREFILL_SECS_PER_100K: f64 = 1.2;
/// 热度曲线:`(距该号上次为本线程起跑的时长 / 缓存 TTL, 热度)`,线性插值。
///
/// 热度 = 回到该号能省下的冷热差价比例 = (同号续轮命中率 − 陌生号命中率) / (1 − 陌生号命中率)。
/// 2026-09-30..10-03 约 25 万次同号续轮实测(TTL=300s 时):30 秒内命中 93%、2 分钟 84%、
/// 4–5 分钟 67–68%、5–5.5 分钟骤降到 43%、10–30 分钟稳定在约 33%;陌生号约 20%
/// (同一客户的 system/工具定义在别的会话里缓存过)。上游缓存寿命就是这个 TTL,故按它伸缩。
const WARMTH_CURVE: [(f64, f64); 6] = [(0.0, 0.92), (0.4, 0.80), (1.0, 0.58), (1.1, 0.28), (2.0, 0.18), (6.0, 0.12)];
/// 记住「线程在哪些号上跑过」的时长,以缓存 TTL 为单位(默认 6 × 300s = 30 分钟,与旧版
/// `affinity_ttl` 默认值一致)。超过即遗忘,热度视为 0。
pub(crate) const MEMORY_TTLS: f64 = 6.0;
/// 一次 429 的重试耗时估计(秒):节流/冷却 + 重新选号 + 再发一次。
pub(crate) const T_RETRY_SECS: f64 = 3.0;
/// 等待切片:决策为「等」时最多睡这么久就回来重新评估(滚动时域)。
pub(crate) const WAIT_SLICE: Duration = Duration::from_millis(500);
/// 在途请求剩余时长的下限(估计再乐观也至少等这么久才可能腾出位子)。
pub(crate) const RESIDUAL_FLOOR: Duration = Duration::from_millis(500);
/// 已跑时长超过中位数后,剩余时长按「已跑时长 × 本系数」估(耗时重尾:跑得越久的请求
/// 往往还要跑更久)。
pub(crate) const HEAVY_TAIL_FACTOR: f64 = 0.5;
/// 同一个号上同时在等的请求数上限;满了就不再把这个号当「可等」候选。
pub(crate) const MAX_WAITERS_PER_ACCOUNT: u32 = 32;
/// 429 聚集:本号 20 秒内 / 20–300 秒内 429 过,优势比分别乘 e^1.08 / e^0.86
/// (联合 logistic 拟合,约 10 万次个人号上游尝试)。
const LOGODDS_429_WITHIN_20S: f64 = 1.08;
const LOGODDS_429_WITHIN_300S: f64 = 0.86;
/// 池级 429 率的时间衰减半衰期与先验(伪计数)。上游 429 率随时间大幅摆动
/// (同一个号几分钟内 1%–17%),短半衰期才跟得住;先验防冷启动时一次 429 就把率拉满。
const POOL_HALF_LIFE: Duration = Duration::from_secs(600);
const POOL_PRIOR_WEIGHT: f64 = 200.0;
const POOL_PRIOR_PERSONAL: f64 = 0.0185;
const POOL_PRIOR_ENTERPRISE: f64 = 0.002;
/// 耗时中位数估计的初值(秒)与 EWMA 系数(对 ln 耗时做,近似几何均值 ≈ 对数正态中位数)。
const DUR_INIT_SECS: f64 = 10.0;
const DUR_EWMA_ALPHA: f64 = 0.05;
/// 线程热度表的懒清扫间隔。
const SWEEP_EVERY: Duration = Duration::from_secs(30);
/// 单线程记住的热号数上限(一个线程同时热在很多号上本身就说明在乱跳)。
const MAX_TOUCHES_PER_THREAD: usize = 8;

/// 号型计价档位。企业号每积分更贵(由 [`CiParams::enterprise_price`] 给出倍数)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PriceClass {
    Personal,
    Enterprise,
}

impl PriceClass {
    fn idx(self) -> usize {
        match self {
            PriceClass::Personal => 0,
            PriceClass::Enterprise => 1,
        }
    }
}

/// 最优控制调度器参数(由 [`SchedulerConfig`] 注入,随 update_tuning 热更;已钳位)。
#[derive(Debug, Clone, Copy)]
pub(crate) struct CiParams {
    pub kappa: f64,
    pub eta: f64,
    pub enterprise_price: f64,
    pub start_spacing: Duration,
    pub pace: Duration,
    pub pace_max_strikes: u32,
    pub pace_personal: bool,
    pub max_wait: Duration,
    pub cache_ttl: Duration,
}

impl CiParams {
    pub fn from_config(c: &SchedulerConfig) -> Self {
        // 钳位与 admin PUT 的范围校验一致;这里再兜一次,防 YAML 基线写出荒谬值。
        let finite = |v: f64, lo: f64, hi: f64, dflt: f64| {
            if v.is_finite() { v.clamp(lo, hi) } else { dflt }
        };
        Self {
            kappa: finite(c.cost_index_kappa, 0.0, 100.0, 1.0),
            eta: finite(c.cost_index_eta, 0.0, 1000.0, 3.0),
            enterprise_price: finite(c.cost_index_enterprise_price, 0.01, 100.0, 3.0),
            start_spacing: Duration::from_millis(c.cost_index_start_spacing_ms.min(10_000)),
            pace: Duration::from_millis(c.cost_index_pace_ms.min(60_000)),
            pace_max_strikes: c.cost_index_pace_max_strikes,
            pace_personal: c.cost_index_pace_personal,
            // 必须远小于 worker 180s 的请求重试时限。
            max_wait: Duration::from_millis(c.cost_index_max_wait_ms.min(120_000)),
            cache_ttl: Duration::from_secs(c.cost_index_cache_ttl_secs.clamp(1, 3600)),
        }
    }

    pub fn price(&self, class: PriceClass) -> f64 {
        match class {
            PriceClass::Personal => 1.0,
            PriceClass::Enterprise => self.enterprise_price,
        }
    }
}

/// 一个候选号的快照(调度器在锁内构造)。
#[derive(Debug, Clone)]
pub(crate) struct Cand {
    pub rank: i64,
    pub class: PriceClass,
    /// 本线程回到该号的热度(见 [`warmth`]):0 = 从没在这里跑过或已遗忘。
    pub warmth: f64,
    /// 预计还要等多久才能在该号上起跑;0 = 此刻就能起跑。
    pub wait: Duration,
    /// 在该号上起跑后撞 429 的概率估计。
    pub p429: f64,
    /// 近期以该号为「家」的活跃线程数(平局时越少越好,见 [`CiState::active_threads`])。
    pub active: u32,
    /// 在途请求数(平局时的次级排序:越空越好)。
    pub inflight: usize,
    /// 最后被选中的时刻(平局时再按最久未用排,与旧版 LRU 相容)。
    pub last_selected: Option<Instant>,
}

/// 距该号上次为本线程起跑 `age` 后,回到该号的热度(见 [`WARMTH_CURVE`])。
/// 超过记忆时长(`MEMORY_TTLS` 个 TTL)返回 0。
pub(crate) fn warmth(age: Duration, ttl: Duration) -> f64 {
    let r = age.as_secs_f64() / ttl.as_secs_f64().max(1e-3);
    if r > MEMORY_TTLS {
        return 0.0;
    }
    WARMTH_CURVE
        .windows(2)
        .find(|seg| r <= seg[1].0)
        .map_or(WARMTH_CURVE[WARMTH_CURVE.len() - 1].1, |seg| {
            let ((r0, w0), (r1, w1)) = (seg[0], seg[1]);
            w0 + (w1 - w0) * (r - r0) / (r1 - r0)
        })
}

/// 预期代价(积分当量)。冷的那部分 `(1-w)` 同时付积分差与冷预填多出的时间。
pub(crate) fn index(c: &Cand, x: u64, p: &CiParams) -> f64 {
    let w = c.warmth.clamp(0.0, 1.0);
    let x5 = x as f64 / 1e5;
    p.price(c.class) * (CREDIT_BASE + x5 * (w * RATE_WARM + (1.0 - w) * RATE_COLD))
        + p.kappa * (c.wait.as_secs_f64() + (1.0 - w) * COLD_PREFILL_SECS_PER_100K * x5)
        + c.p429 * (p.kappa * T_RETRY_SECS + p.eta)
}

/// 决策结果。`Go(i)` 的 i 是传入切片的下标。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    Go(usize),
    /// 最优的是一个要等的号:睡这么久(≤ [`WAIT_SLICE`])再重新评估。附带该号下标,
    /// 调用方据此登记等待者。
    Wait(usize, Duration),
    Nothing,
}

/// 选指数最小的候选。`immediate_only` = 等待预算已用完,只在能立刻起跑的号里挑。
/// 平局依次比:等待短 → 更热 → 近期活跃线程少 → 在途少 → 最久未选(None 最先)→ 下标。
pub(crate) fn decide(cands: &[Cand], x: u64, p: &CiParams, immediate_only: bool) -> Decision {
    let best = cands
        .iter()
        .enumerate()
        .filter(|(_, c)| !immediate_only || c.wait.is_zero())
        .map(|(i, c)| (i, index(c, x, p), c))
        .min_by(|(ia, sa, a), (ib, sb, b)| {
            sa.total_cmp(sb)
                .then(a.wait.cmp(&b.wait))
                .then(b.warmth.total_cmp(&a.warmth))
                .then(a.active.cmp(&b.active))
                .then(a.inflight.cmp(&b.inflight))
                .then(match (a.last_selected, b.last_selected) {
                    (None, None) => std::cmp::Ordering::Equal,
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (Some(x), Some(y)) => x.cmp(&y),
                })
                .then(ia.cmp(ib))
        });
    match best {
        None => Decision::Nothing,
        Some((i, _, c)) if c.wait.is_zero() => Decision::Go(i),
        Some((i, _, c)) => Decision::Wait(i, c.wait.min(WAIT_SLICE)),
    }
}

/// 在该号上撞 429 的概率:池级率按本号近期 429 在优势比上加成。
pub(crate) fn p429(pool_rate: f64, since_last_429: Option<Duration>) -> f64 {
    let p = pool_rate.clamp(1e-6, 1.0 - 1e-6);
    let boost = match since_last_429 {
        Some(d) if d <= Duration::from_secs(20) => LOGODDS_429_WITHIN_20S,
        Some(d) if d <= Duration::from_secs(300) => LOGODDS_429_WITHIN_300S,
        _ => 0.0,
    };
    let odds = p / (1.0 - p) * boost.exp();
    odds / (1.0 + odds)
}

/// 单个在途请求的剩余时长估计。
pub(crate) fn residual(elapsed: Duration, dmed: Duration) -> Duration {
    let r = if elapsed < dmed {
        dmed - elapsed
    } else {
        elapsed.mul_f64(HEAVY_TAIL_FACTOR)
    };
    r.max(RESIDUAL_FLOOR)
}

/// 号并发已满时,预计多久轮到本请求:在途剩余时长排序后取第 k 个(k = 已在等的请求数)。
/// 等待者多于在途数时,按「每轮一个中位时长」往后顺延。
pub(crate) fn busy_wait(inflight_elapsed: &[Duration], waiters_ahead: u32, dmed: Duration) -> Duration {
    if inflight_elapsed.is_empty() {
        // 并发满却看不到在途记录(并发上限刚被调小、旧许可还在旧信号量上):保守估一个中位时长。
        return dmed.max(RESIDUAL_FLOOR);
    }
    let mut r: Vec<Duration> = inflight_elapsed.iter().map(|e| residual(*e, dmed)).collect();
    r.sort();
    let n = r.len();
    let k = waiters_ahead as usize;
    r[k % n] + dmed * (k / n) as u32
}

/// 按时间衰减的比率估计(分子 / 分母各自指数衰减,再与先验按伪计数混合)。
#[derive(Debug, Clone)]
pub(crate) struct DecayedRate {
    hits: f64,
    total: f64,
    at: Option<Instant>,
    prior: f64,
}

impl DecayedRate {
    fn new(prior: f64) -> Self {
        Self { hits: 0.0, total: 0.0, at: None, prior }
    }

    fn decay_to(&mut self, now: Instant) {
        if let Some(at) = self.at {
            let dt = now.saturating_duration_since(at).as_secs_f64();
            let f = 0.5f64.powf(dt / POOL_HALF_LIFE.as_secs_f64());
            self.hits *= f;
            self.total *= f;
        }
        self.at = Some(now);
    }

    pub fn rate(&mut self, now: Instant) -> f64 {
        self.decay_to(now);
        (self.hits + self.prior * POOL_PRIOR_WEIGHT) / (self.total + POOL_PRIOR_WEIGHT)
    }

    pub fn note(&mut self, now: Instant, attempt: bool, hit: bool) {
        self.decay_to(now);
        if attempt {
            self.total += 1.0;
        }
        if hit {
            self.hits += 1.0;
        }
    }
}

/// 一条线程(会话 + 线程指纹)的缓存热度:在哪些号上、最后一次成功服务的时刻。
#[derive(Debug, Clone)]
struct ThreadWarm {
    touches: Vec<(String, Instant)>,
    last_input: u64,
    last_seen: Instant,
}

/// 最优控制调度器的共享状态(调度器里一把独立的 `Mutex`,锁序 `entries → ci`)。
#[derive(Debug)]
pub(crate) struct CiState {
    warm: HashMap<String, ThreadWarm>,
    last_sweep: Instant,
    pool: [DecayedRate; 2],
    ln_dur: f64,
    /// 号 → 近一个 TTL 内以它为「家」(最近一次跑在它上面)的线程数。清扫时重算
    /// (至多滞后 [`SWEEP_EVERY`]);两次清扫之间新派到冷号的线程由 [`Self::note_new_home`] 先记上。
    active: HashMap<String, u32>,
}

impl CiState {
    pub fn new(now: Instant) -> Self {
        Self {
            warm: HashMap::new(),
            last_sweep: now,
            pool: [
                DecayedRate::new(POOL_PRIOR_PERSONAL),
                DecayedRate::new(POOL_PRIOR_ENTERPRISE),
            ],
            ln_dur: DUR_INIT_SECS.ln(),
            active: HashMap::new(),
        }
    }

    /// 距该号上次为本线程起跑多久;没跑过或已超过记忆时长返回 `None`。
    pub fn touch_age(&self, key: &str, account: &str, now: Instant, ttl: Duration) -> Option<Duration> {
        let memory = ttl.mul_f64(MEMORY_TTLS);
        self.warm.get(key).and_then(|w| {
            w.touches
                .iter()
                .find(|(a, _)| a == account)
                .map(|(_, t)| now.saturating_duration_since(*t))
                .filter(|age| *age <= memory)
        })
    }

    /// 本线程在该号上是否仍在缓存 TTL 内(计数与测试用;选号看 [`warmth`])。
    pub fn is_warm(&self, key: &str, account: &str, now: Instant, ttl: Duration) -> bool {
        self.touch_age(key, account, now, ttl).is_some_and(|age| age <= ttl)
    }

    /// 近期以该号为家的活跃线程数(见 `active` 字段)。
    pub fn active_threads(&self, account: &str) -> u32 {
        self.active.get(account).copied().unwrap_or(0)
    }

    /// 一个新线程刚被派到冷号 `account`:不等下次清扫,先把它计入该号的活跃线程。
    pub fn note_new_home(&mut self, account: &str) {
        *self.active.entry(account.to_string()).or_insert(0) += 1;
    }

    /// 本线程上次实测的输入 token(0 = 未知)。
    pub fn thread_input(&self, key: &str) -> u64 {
        self.warm.get(key).map_or(0, |w| w.last_input)
    }

    /// 记一次成功服务:该线程在 `account` 上从 `at` 起算热。
    pub fn touch(&mut self, key: &str, account: &str, at: Instant, input_tokens: Option<u64>, now: Instant) {
        let w = self.warm.entry(key.to_string()).or_insert_with(|| ThreadWarm {
            touches: Vec::new(),
            last_input: 0,
            last_seen: now,
        });
        w.last_seen = now;
        if let Some(x) = input_tokens {
            w.last_input = x;
        }
        match w.touches.iter_mut().find(|(a, _)| a == account) {
            Some(slot) => slot.1 = slot.1.max(at),
            None => w.touches.push((account.to_string(), at)),
        }
        if w.touches.len() > MAX_TOUCHES_PER_THREAD {
            w.touches.sort_by(|a, b| b.1.cmp(&a.1));
            w.touches.truncate(MAX_TOUCHES_PER_THREAD);
        }
    }

    /// 懒清扫:每 [`SWEEP_EVERY`] 最多一次,丢掉超过记忆时长的线程,并重算各号的活跃线程数。
    pub fn maybe_sweep(&mut self, now: Instant, ttl: Duration) {
        if now.saturating_duration_since(self.last_sweep) < SWEEP_EVERY {
            return;
        }
        self.last_sweep = now;
        let keep = ttl.mul_f64(MEMORY_TTLS) + SWEEP_EVERY;
        self.warm
            .retain(|_, w| now.saturating_duration_since(w.last_seen) <= keep);
        self.active.clear();
        for w in self.warm.values() {
            if now.saturating_duration_since(w.last_seen) > ttl {
                continue;
            }
            if let Some((home, _)) = w.touches.iter().max_by_key(|(_, t)| *t) {
                *self.active.entry(home.clone()).or_insert(0) += 1;
            }
        }
    }

    pub fn threads(&self) -> usize {
        self.warm.len()
    }

    /// 测试用:把所有时间戳整体往过去挪 `d`,模拟时间流逝(Instant 不能快进)。
    #[cfg(test)]
    pub fn age_for_test(&mut self, d: Duration) {
        self.last_sweep -= d;
        for w in self.warm.values_mut() {
            w.last_seen -= d;
            for t in w.touches.iter_mut() {
                t.1 -= d;
            }
        }
    }

    /// 测试用:只把一条线程的时间戳往过去挪 `d`。
    #[cfg(test)]
    pub fn age_thread_for_test(&mut self, key: &str, d: Duration) {
        if let Some(w) = self.warm.get_mut(key) {
            w.last_seen -= d;
            for t in w.touches.iter_mut() {
                t.1 -= d;
            }
        }
    }

    pub fn pool_rate(&mut self, class: PriceClass, now: Instant) -> f64 {
        self.pool[class.idx()].rate(now)
    }

    /// 记一次上游尝试(`attempt`)与/或一次 429(`hit`)。
    pub fn note_pool(&mut self, class: PriceClass, now: Instant, attempt: bool, hit: bool) {
        self.pool[class.idx()].note(now, attempt, hit);
    }

    /// 成功请求耗时中位数的估计。
    pub fn dmed(&self) -> Duration {
        Duration::from_secs_f64(self.ln_dur.exp().clamp(0.5, 600.0))
    }

    pub fn note_duration(&mut self, d: Duration) {
        let secs = d.as_secs_f64().max(0.05);
        self.ln_dur = (1.0 - DUR_EWMA_ALPHA) * self.ln_dur + DUR_EWMA_ALPHA * secs.ln();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> CiParams {
        CiParams::from_config(&SchedulerConfig::default())
    }

    /// `warm` = 刚在该号上跑过(热度取曲线起点),否则完全冷。
    fn cand(warm: bool, wait_ms: u64) -> Cand {
        Cand {
            rank: 100,
            class: PriceClass::Personal,
            warmth: if warm { warmth(Duration::ZERO, Duration::from_secs(300)) } else { 0.0 },
            wait: Duration::from_millis(wait_ms),
            p429: 0.0,
            active: 0,
            inflight: 0,
            last_selected: None,
        }
    }

    #[test]
    fn index_matches_calibrated_credit_model() {
        let p = params();
        // x = 10 万,完全热(热度 1):0.086+0.342 = 0.428,不付冷预填时间。
        let mut hot = cand(true, 0);
        hot.warmth = 1.0;
        assert!((index(&hot, 100_000, &p) - 0.428).abs() < 1e-9);
        // 完全冷:0.086+1.197 = 1.283 积分,外加 κ × 冷预填多出的 1.2 秒。
        assert!((index(&cand(false, 0), 100_000, &p) - (1.283 + 1.2)).abs() < 1e-9);
        // 企业号单价 3 倍(只乘积分,时间按 κ 计)。
        let mut ent = cand(false, 0);
        ent.class = PriceClass::Enterprise;
        assert!((index(&ent, 100_000, &p) - (3.0 * 1.283 + 1.2)).abs() < 1e-9);
        // 热度 0.5:积分与冷预填时间都按一半计。
        let mut half = cand(false, 0);
        half.warmth = 0.5;
        let expect = 0.086 + 0.5 * 0.342 + 0.5 * 1.197 + 0.5 * 1.2;
        assert!((index(&half, 100_000, &p) - expect).abs() < 1e-9);
        // 等待按 κ 计价、429 按 κ·t_retry + η 计价。
        let mut c = hot.clone();
        c.wait = Duration::from_secs(2);
        c.p429 = 0.1;
        let expect = 0.428 + 1.0 * 2.0 + 0.1 * (1.0 * T_RETRY_SECS + 3.0);
        assert!((index(&c, 100_000, &p) - expect).abs() < 1e-9);
    }

    /// 热号值得等的上限 W* = 热度 × x/1e5 × (Δr·单价/κ + 冷预填秒数)(个人号)。
    /// 2026-10-03 修正前漏了冷预填这一项,20 万上下文只肯等约 1.7 秒,而冷启动本身就要
    /// 多等 2–3 秒首包 —— 换号又贵又不快。
    #[test]
    fn wait_threshold_counts_cold_prefill_time() {
        let dr = RATE_COLD - RATE_WARM;
        let w0 = warmth(Duration::ZERO, Duration::from_secs(300));
        for (kappa, x) in [(1.0, 400_000u64), (1.0, 200_000), (1.0, 30_000), (0.1, 100_000), (0.3, 200_000)] {
            let p = CiParams { kappa, ..params() };
            let w_star = w0 * x as f64 / 1e5 * (dr / kappa + COLD_PREFILL_SECS_PER_100K);
            // 略短于 W*:等热号更优;略长于 W*:立刻去冷号更优。
            let just_under = [cand(true, ((w_star - 0.05) * 1000.0) as u64), cand(false, 0)];
            let just_over = [cand(true, ((w_star + 0.05) * 1000.0) as u64), cand(false, 0)];
            assert!(matches!(decide(&just_under, x, &p, false), Decision::Wait(0, _)), "κ={kappa} x={x} 应等");
            assert_eq!(decide(&just_over, x, &p, false), Decision::Go(1), "κ={kappa} x={x} 应换冷号");
        }
        // 对照修正前:κ=1、20 万上下文,W* 从 1.71 秒提到约 3.8 秒。
        let w_star = w0 * 2.0 * (dr + COLD_PREFILL_SECS_PER_100K);
        assert!(w_star > 3.5 && w_star < 4.0, "W*={w_star}");
    }

    #[test]
    fn warmth_curve_follows_measured_cache_lifetime() {
        let ttl = Duration::from_secs(300);
        let at = |s: u64| warmth(Duration::from_secs(s), ttl);
        // 刚跑过最热;单调不增。
        assert!((at(0) - 0.92).abs() < 1e-9);
        let mut prev = at(0);
        for s in (0..=1800).step_by(15) {
            let w = at(s);
            assert!(w <= prev + 1e-12 && w > 0.0, "{s}s 处热度 {w} 应单调不增且为正");
            prev = w;
        }
        // 上游缓存约 5 分钟过期:越过 TTL 骤降,但回原号仍比全新的号强(实测约 33% 对 20%)。
        assert!(at(299) > 0.55 && at(330) < 0.3 && at(600) > 0.1);
        // 超过记忆时长(6 个 TTL)就忘掉。
        assert!(at(1800) > 0.0);
        assert_eq!(at(1801), 0.0);
        // 曲线按 TTL 等比伸缩:运维把 TTL 调成 600 秒,10 分钟前跑过的号与默认下 5 分钟前一样热。
        let long = Duration::from_secs(600);
        assert!((warmth(Duration::from_secs(600), long) - at(300)).abs() < 1e-9);
    }

    #[test]
    fn decide_prefers_warmer_then_less_crowded_accounts_on_ties() {
        let p = params();
        // 两个完全冷的候选,代价相同:先比挂着的活跃线程数(少的优先),再比在途、最久未选。
        let now = Instant::now();
        let mut busy_home = cand(false, 0);
        busy_home.active = 5;
        busy_home.last_selected = Some(now - Duration::from_secs(60));
        let mut quiet = cand(false, 0);
        quiet.active = 1;
        quiet.last_selected = Some(now);
        assert_eq!(decide(&[busy_home, quiet], 50_000, &p, false), Decision::Go(1), "近期活跃会话少的号优先");
    }

    #[test]
    fn decide_caps_wait_slice_and_respects_immediate_only() {
        let p = params();
        let cs = [cand(true, 3000), cand(false, 0)];
        // x = 100 万:热号值得等 3s。
        assert_eq!(decide(&cs, 1_000_000, &p, false), Decision::Wait(0, WAIT_SLICE));
        // 预算用完:只看能立刻起跑的。
        assert_eq!(decide(&cs, 1_000_000, &p, true), Decision::Go(1));
        assert_eq!(decide(&[cand(true, 100)], 1_000_000, &p, true), Decision::Nothing);
        assert_eq!(decide(&[], 1, &p, false), Decision::Nothing);
    }

    #[test]
    fn decide_tie_breaks_by_inflight_then_lru() {
        let p = params();
        let now = Instant::now();
        let mut a = cand(false, 0);
        let mut b = cand(false, 0);
        a.inflight = 2;
        b.inflight = 1;
        assert_eq!(decide(&[a.clone(), b.clone()], 50_000, &p, false), Decision::Go(1));
        a.inflight = 1;
        a.last_selected = Some(now);
        b.last_selected = Some(now - Duration::from_secs(5));
        assert_eq!(decide(&[a.clone(), b.clone()], 50_000, &p, false), Decision::Go(1), "同载按最久未选");
        b.last_selected = None;
        a.last_selected = None;
        assert_eq!(decide(&[a, b], 50_000, &p, false), Decision::Go(0), "全平按下标");
    }

    #[test]
    fn p429_boosts_recent_429_at_boundaries() {
        let base = 0.02;
        assert!((p429(base, None) - base).abs() < 1e-12);
        assert!((p429(base, Some(Duration::from_secs(301))) - base).abs() < 1e-12);
        let o = |p: f64| p / (1.0 - p);
        let r20 = p429(base, Some(Duration::from_secs(20)));
        let r300 = p429(base, Some(Duration::from_secs(300)));
        assert!((o(r20) / o(base) - LOGODDS_429_WITHIN_20S.exp()).abs() < 1e-9);
        assert!((o(r300) / o(base) - LOGODDS_429_WITHIN_300S.exp()).abs() < 1e-9);
        assert!(r20 > r300 && r300 > base);
    }

    #[test]
    fn decayed_rate_uses_prior_and_half_life() {
        let t0 = Instant::now();
        let mut r = DecayedRate::new(0.02);
        assert!((r.rate(t0) - 0.02).abs() < 1e-12, "无观测 = 先验");
        for _ in 0..200 {
            r.note(t0, true, true);
        }
        // 200 次全 429 与 200 伪计数先验各半。
        assert!((r.rate(t0) - (200.0 + 4.0) / 400.0).abs() < 1e-9);
        // 一个半衰期后观测减半。
        let later = t0 + POOL_HALF_LIFE;
        assert!((r.rate(later) - (100.0 + 4.0) / 300.0).abs() < 1e-9);
    }

    #[test]
    fn busy_wait_orders_residuals_and_queues_waiters() {
        let dmed = Duration::from_secs(10);
        // 已跑 2s、8s、30s:剩余 8s、2s、15s(重尾)。
        let inflight = [Duration::from_secs(2), Duration::from_secs(8), Duration::from_secs(30)];
        assert_eq!(busy_wait(&inflight, 0, dmed), Duration::from_secs(2));
        assert_eq!(busy_wait(&inflight, 1, dmed), Duration::from_secs(8));
        assert_eq!(busy_wait(&inflight, 2, dmed), Duration::from_secs(15));
        // 等待者比在途多:顺延一个中位时长。
        assert_eq!(busy_wait(&inflight, 3, dmed), Duration::from_secs(12));
        assert_eq!(busy_wait(&[], 0, dmed), dmed);
        assert_eq!(residual(Duration::from_millis(9_900), dmed), RESIDUAL_FLOOR);
    }

    #[test]
    fn warm_table_touch_ttl_and_sweep() {
        let t0 = Instant::now();
        let ttl = Duration::from_secs(300);
        let mut s = CiState::new(t0);
        assert!(!s.is_warm("k", "a", t0, ttl));
        assert_eq!(s.touch_age("k", "a", t0, ttl), None);
        s.touch("k", "a", t0, Some(120_000), t0);
        assert!(s.is_warm("k", "a", t0 + Duration::from_secs(299), ttl));
        assert!(!s.is_warm("k", "a", t0 + Duration::from_secs(301), ttl));
        assert!(!s.is_warm("k", "b", t0, ttl));
        assert_eq!(s.thread_input("k"), 120_000);
        // 过了 TTL 仍记得它在哪个号上跑过(回原号还有机会命中),直到记忆时长。
        assert_eq!(
            s.touch_age("k", "a", t0 + Duration::from_secs(900), ttl),
            Some(Duration::from_secs(900))
        );
        assert_eq!(s.touch_age("k", "a", t0 + Duration::from_secs(1801), ttl), None);
        // 旧的触达时刻不会覆盖新的。
        s.touch("k", "a", t0 - Duration::from_secs(100), None, t0);
        assert!(s.is_warm("k", "a", t0 + Duration::from_secs(299), ttl));
        assert_eq!(s.thread_input("k"), 120_000, "None 不清掉已知输入量");
        // 记忆时长内不清扫;超过才清。
        s.maybe_sweep(t0 + Duration::from_secs(10), ttl);
        assert_eq!(s.threads(), 1, "未到清扫间隔不清");
        s.maybe_sweep(t0 + ttl + SWEEP_EVERY + Duration::from_secs(1), ttl);
        assert_eq!(s.threads(), 1, "过了 TTL 但还在记忆时长内,不清");
        s.maybe_sweep(t0 + ttl.mul_f64(MEMORY_TTLS) + SWEEP_EVERY + Duration::from_secs(1), ttl);
        assert_eq!(s.threads(), 0);
    }

    #[test]
    fn active_threads_count_recent_homes_per_account() {
        let t0 = Instant::now();
        let ttl = Duration::from_secs(300);
        let mut s = CiState::new(t0);
        // t1 先在 a、后在 b 跑过 → 归 b;t2 在 a;t3 很久以前在 a(不算活跃)。
        s.touch("t1", "a", t0, None, t0);
        s.touch("t1", "b", t0 + Duration::from_secs(10), None, t0 + Duration::from_secs(10));
        s.touch("t2", "a", t0 + Duration::from_secs(20), None, t0 + Duration::from_secs(20));
        s.touch("t3", "a", t0, None, t0);
        s.age_thread_for_test("t3", Duration::from_secs(400));
        let now = t0 + SWEEP_EVERY + Duration::from_secs(30);
        s.maybe_sweep(now, ttl);
        assert_eq!(s.active_threads("a"), 1);
        assert_eq!(s.active_threads("b"), 1);
        assert_eq!(s.active_threads("c"), 0);
        // 新线程刚被派到 c:清扫前就先计上,免得同一窗口里的新线程全挤到 c。
        s.note_new_home("c");
        assert_eq!(s.active_threads("c"), 1);
    }

    #[test]
    fn duration_ewma_moves_toward_observations() {
        let mut s = CiState::new(Instant::now());
        assert!((s.dmed().as_secs_f64() - DUR_INIT_SECS).abs() < 1e-9);
        for _ in 0..200 {
            s.note_duration(Duration::from_secs(20));
        }
        assert!((s.dmed().as_secs_f64() - 20.0).abs() < 0.1);
    }

    #[test]
    fn params_are_clamped() {
        let mut c = SchedulerConfig::default();
        c.cost_index_kappa = f64::NAN;
        c.cost_index_enterprise_price = 0.0;
        c.cost_index_max_wait_ms = 10_000_000;
        c.cost_index_cache_ttl_secs = 0;
        let p = CiParams::from_config(&c);
        assert_eq!(p.kappa, 1.0);
        assert_eq!(p.enterprise_price, 0.01);
        assert_eq!(p.max_wait, Duration::from_secs(120));
        assert_eq!(p.cache_ttl, Duration::from_secs(1));
    }
}

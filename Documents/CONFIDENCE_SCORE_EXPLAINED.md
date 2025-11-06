# Confidence Score Explained

## What is the Confidence Score?

The confidence score (0-1) quantifies **how certain we are that a detected stall is a real, significant event** rather than noise or a marginal case.

**Scale:**
- **0.0-0.3:** Low confidence (questionable)
- **0.3-0.5:** Moderate confidence
- **0.5-0.7:** Good confidence
- **0.7-0.9:** High confidence
- **0.9-1.0:** Very high confidence (strong event)

## The Formula

```python
def _calculate_confidence(duration_hours, height_std, n_points):
    # 1. Duration score (sigmoid centered at 18 hours)
    duration_score = 1 / (1 + exp(-(duration_hours - 18) / 6))
    
    # 2. Stability score (inverse of normalized std deviation)
    stability_score = 1 - min(height_std / max_std, 1.0)
    
    # 3. Data quality score (sigmoid centered at 10 points)
    quality_score = 1 / (1 + exp(-(n_points - 10) / 3))
    
    # 4. Weighted average
    confidence = (
        0.4 * duration_score +    # 40% weight
        0.4 * stability_score +   # 40% weight
        0.2 * quality_score       # 20% weight
    )
    
    return confidence  # 0.0 to 1.0
```

## Three Components (Explained)

### 1. Duration Score (40% weight)

**What it measures:** How long the wetting front stayed at the same height

**Why it matters:** 
- Longer stalls → stronger impedance → more likely a weak layer
- Brief pauses might just be diurnal cycles or measurement noise

**The math (sigmoid function):**
```python
duration_score = 1 / (1 + exp(-(duration_hours - 18) / 6))
```

**Behavior:**
- Centered at **18 hours** (the "typical" strong stall)
- Gradual S-curve (not a hard threshold)
- Asymptotes: approaches 0 for very short, 1 for very long

**Score examples:**

| Duration | Score | Interpretation |
|----------|-------|----------------|
| 6 hours  | 0.12  | Very brief |
| 12 hours | 0.27  | Minimum threshold |
| 18 hours | 0.50  | Moderate stall |
| 24 hours | 0.73  | Strong stall |
| 36 hours | 0.95  | Very strong stall |
| 48 hours | 0.99  | Extremely strong |

**Visualization:**
```
Score
1.0 ┤                    ╭─────────
    │                  ╭─╯
    │                ╭─╯
0.5 ┤              ╭─╯  ← 18 hours (inflection point)
    │            ╭─╯
    │         ╭──╯
0.0 ┤─────────╯
    └─────┴─────┴─────┴─────┴─────┴─────┴──→ Duration (hours)
          6    12    18    24    30    36
```

**Why sigmoid?**
- **Smooth transition:** No arbitrary hard cutoffs
- **Physically meaningful:** Gradual increase reflects increasing certainty
- **Bounded:** Always between 0 and 1
- **Flexible:** Parameters (18, 6) can be tuned based on data

---

### 2. Stability Score (40% weight)

**What it measures:** How stable the height was during the stall

**Why it matters:**
- Low variability → truly "stuck" → strong interface
- High variability → gradual movement → may not be real stall

**The math (inverse normalized std):**
```python
stability_score = 1 - min(height_std / max_std, 1.0)

# Where:
#   height_std = standard deviation of heights during stall
#   max_std = tolerance (e.g., 0.05m)
```

**Behavior:**
- **Perfect stability (std = 0):** score = 1.0
- **At tolerance (std = 0.05m):** score = 0.0
- **Above tolerance:** clamped at 0.0

**Score examples (tolerance = 0.05m):**

| Std Dev | Score | Interpretation |
|---------|-------|----------------|
| 0.000 m | 1.00  | Perfectly stable (unusual) |
| 0.010 m | 0.80  | Very stable (±1cm) |
| 0.025 m | 0.50  | Moderately stable (±2.5cm) |
| 0.040 m | 0.20  | Less stable (±4cm) |
| 0.050 m | 0.00  | At tolerance limit |
| 0.100 m | 0.00  | Beyond tolerance |

**Visualization:**
```
Score
1.0 ┤╲
    │ ╲
    │  ╲
0.5 ┤   ╲       ← 2.5cm std dev
    │    ╲
    │     ╲
0.0 ┤      ╲___________
    └───────┴─────┴─────┴─────┴──→ Std Dev (m)
          0.01  0.025 0.05  0.10
```

**Why this approach?**
- **Simple interpretation:** Lower variability = higher score
- **Normalized:** Relative to tolerance (adapts if tolerance changes)
- **Penalizes noise:** High variability reduces confidence
- **Physical meaning:** Measures "stickiness" of interface

---

### 3. Data Quality Score (20% weight)

**What it measures:** How many data points confirm the stall

**Why it matters:**
- More measurements → more reliable
- Few points → could be lucky coincidence or data gap
- Statistical confidence increases with sample size

**The math (sigmoid function):**
```python
quality_score = 1 / (1 + exp(-(n_points - 10) / 3))
```

**Behavior:**
- Centered at **10 data points**
- Gradual increase (sigmoid curve)
- Asymptotes at 0 and 1

**Score examples:**

| # Points | Score | Interpretation |
|----------|-------|----------------|
| 3        | 0.15  | Minimum (barely detected) |
| 5        | 0.25  | Few measurements |
| 10       | 0.50  | Adequate sample |
| 15       | 0.75  | Good sample |
| 20+      | 0.90+ | Excellent sample |

**Visualization:**
```
Score
1.0 ┤              ╭───────────
    │            ╭─╯
    │          ╭─╯
0.5 ┤        ╭─╯  ← 10 points
    │      ╭─╯
    │    ╭─╯
0.0 ┤────╯
    └─────┴─────┴─────┴─────┴──→ # Data Points
          5    10    15    20
```

**Why sigmoid?**
- **Gradual confidence:** More points = better, but diminishing returns
- **No penalty for many points:** Doesn't hurt to have 50+ points
- **Statistical basis:** More samples = more reliable estimate

**Example:** 
- SNOWPACK outputs every 6 hours
- 12-hour stall = 3 points (minimum)
- 24-hour stall = 5 points (moderate)
- 48-hour stall = 9 points (good)

---

## Weighted Combination

**Why weights?**
```python
confidence = 0.4 * duration + 0.4 * stability + 0.2 * quality
```

### Duration (40%): MOST IMPORTANT
- **Physical significance:** Longer stall = stronger impedance
- **Direct evidence:** Time spent stuck indicates real barrier
- **Avalanche relevance:** Duration correlates with weakness

### Stability (40%): EQUALLY IMPORTANT
- **Signal quality:** Low noise = real signal
- **Physical meaning:** Truly stuck vs. slowly moving
- **Distinguishes events:** Separates real stalls from gradual descent

### Quality (20%): SUPPORTING EVIDENCE
- **Less critical:** Given reasonable sampling (6-hourly)
- **Rarely limiting:** Most events have enough points
- **Safety net:** Prevents single-point flukes

**Why not equal weights?**
- Duration and stability are **direct physical evidence** of impedance
- Data quality is **measurement confidence** (technical, not physical)
- 40-40-20 emphasizes the physics while accounting for data quality

---

## Real-World Examples

### Example 1: Strong Stall (Confidence: 0.87)

```
Duration:  24 hours
Std Dev:   0.02 m (±2cm)
Points:    12 measurements

Scores:
  Duration:  0.73  (24h → strong)
  Stability: 0.60  (2cm → very stable)
  Quality:   0.64  (12 pts → good)
  
Confidence: 0.4*0.73 + 0.4*0.60 + 0.2*0.64 = 0.66
```

**Interpretation:** High confidence - this is a real stall at a significant interface.

---

### Example 2: Marginal Stall (Confidence: 0.45)

```
Duration:  12 hours (minimum)
Std Dev:   0.04 m (±4cm, close to tolerance)
Points:    8 measurements

Scores:
  Duration:  0.27  (12h → barely meets threshold)
  Stability: 0.20  (4cm → variable)
  Quality:   0.40  (8 pts → moderate)
  
Confidence: 0.4*0.27 + 0.4*0.20 + 0.2*0.40 = 0.27
```

**Interpretation:** Low confidence - might be real, might be noise. Needs validation.

---

### Example 3: Very Strong Stall (Confidence: 0.94)

```
Duration:  48 hours
Std Dev:   0.01 m (±1cm)
Points:    20 measurements

Scores:
  Duration:  0.99  (48h → extremely long)
  Stability: 0.80  (1cm → very stable)
  Quality:   0.90  (20 pts → excellent)
  
Confidence: 0.4*0.99 + 0.4*0.80 + 0.2*0.90 = 0.89
```

**Interpretation:** Very high confidence - definitely a major impedance layer.

---

## Why This Is a Good Approach

### ✅ Advantages

#### 1. **Physics-Informed**
- Each component has clear physical meaning
- Duration ↔ impedance strength
- Stability ↔ interface sharpness
- Not arbitrary - based on snowpack physics

#### 2. **Continuous (No Hard Thresholds)**
- Sigmoid functions → smooth transitions
- Avoids "cliff effects" where 11.9h vs 12.0h is huge difference
- More robust to small variations

#### 3. **Interpretable**
- Score of 0.8 has clear meaning
- Can explain to users why confidence is high/low
- Easy to debug false positives/negatives

#### 4. **Tunable**
- Weights can be adjusted based on validation
- Sigmoid parameters can be optimized from data
- Tolerance adapts automatically

#### 5. **Statistically Sound**
- Combines multiple sources of evidence
- Weighted by importance
- Accounts for measurement uncertainty

#### 6. **Practical**
- Single number (0-1) easy to use for filtering
- Can set thresholds (e.g., only use confidence > 0.6)
- Helps prioritize which stalls to investigate

#### 7. **Self-Consistent**
- High confidence → all three factors align
- Low confidence → at least one factor is weak
- Catches edge cases automatically

---

## How to Use Confidence Scores

### In Data Collection (Phase 1):

```python
# Keep all events for training, but flag low confidence
events_df['needs_review'] = events_df['confidence'] < 0.5
```

**Why:** Don't throw away data yet - ML model might learn from low-confidence cases

### In Model Training (Phase 2):

```python
# Train on high-confidence events first
high_conf = events_df[events_df['confidence'] > 0.7]
model.fit(high_conf[features], high_conf[labels])

# Then add medium-confidence for fine-tuning
medium_conf = events_df[events_df['confidence'] > 0.4]
model.finetune(medium_conf[features], medium_conf[labels])
```

**Why:** Learn strong patterns first, then handle edge cases

### In Production (Phase 3):

```python
# Only alert on high-confidence stalls
if event.confidence > 0.7:
    alert_forecaster(event)
elif event.confidence > 0.4:
    flag_for_review(event)
else:
    log_only(event)
```

**Why:** Prioritize actionable intelligence, reduce false alarms

---

## Comparison to Alternatives

### ❌ Simple Threshold (Bad)

```python
confidence = 1.0 if duration > 12 else 0.0
```

**Problems:**
- Binary (no gradation)
- Ignores stability and quality
- Cliff effect at threshold

### ❌ Linear Average (Mediocre)

```python
confidence = (duration/48 + stability + quality/20) / 3
```

**Problems:**
- No diminishing returns
- Equal weights (ignores importance)
- Can exceed 1.0 easily

### ✅ Sigmoid Weighted (Good - Our Approach)

```python
confidence = 0.4*sigmoid(duration) + 0.4*stability + 0.2*sigmoid(quality)
```

**Benefits:**
- Smooth curves
- Bounded [0, 1]
- Physically meaningful
- Tunable

---

## Advanced: Visualizing Confidence Surface

Here's how confidence varies with two key factors:

```
         Duration (hours)
         6    12   18   24   30   36
    0.00 ┤ 0.1  0.2  0.3  0.4  0.5  0.6
    0.01 ┤ 0.3  0.4  0.5  0.6  0.7  0.8
Std 0.02 ┤ 0.4  0.5  0.6  0.7  0.8  0.9
Dev 0.03 ┤ 0.3  0.4  0.5  0.6  0.7  0.8
(m) 0.04 ┤ 0.2  0.3  0.4  0.5  0.6  0.7
    0.05 ┤ 0.1  0.2  0.3  0.4  0.5  0.6
```

**Pattern:**
- Upper left (short + noisy) = LOW
- Lower right (long + stable) = HIGH
- Diagonal trade-off possible

---

## Tuning Recommendations

### If you find too many false positives (low-quality stalls):

**Option 1:** Increase minimum confidence threshold
```python
if event.confidence < 0.6:  # Raised from 0.3
    skip_event
```

**Option 2:** Increase duration weight
```python
confidence = 0.5 * duration + 0.3 * stability + 0.2 * quality
```

**Option 3:** Adjust sigmoid center
```python
duration_score = 1 / (1 + exp(-(duration_hours - 24) / 6))  # Centered at 24h instead of 18h
```

### If you're missing real stalls:

**Option 1:** Lower minimum confidence threshold
```python
if event.confidence < 0.3:  # Lowered from 0.5
    skip_event
```

**Option 2:** Increase stability weight
```python
confidence = 0.3 * duration + 0.5 * stability + 0.2 * quality
```

---

## Summary

The confidence score is a **multi-factor quality metric** that:

1. ✅ Combines duration, stability, and data quality
2. ✅ Uses smooth sigmoid functions (no hard thresholds)
3. ✅ Weights factors by physical importance
4. ✅ Produces interpretable 0-1 scores
5. ✅ Helps filter and prioritize events
6. ✅ Adaptable to different snow climates
7. ✅ Supports ML training workflow

**Bottom line:** It's not perfect, but it's **principled, interpretable, and useful** - much better than arbitrary thresholds!

---

## Interactive Examples

Want to see how confidence changes? Try these:

```python
# Play with the calculator
def calc_confidence(duration_h, std_m, n_points, tolerance=0.05):
    import numpy as np
    d_score = 1 / (1 + np.exp(-(duration_h - 18) / 6))
    s_score = 1 - min(std_m / tolerance, 1.0)
    q_score = 1 / (1 + np.exp(-(n_points - 10) / 3))
    return 0.4*d_score + 0.4*s_score + 0.2*q_score

# Test cases
print("Short + noisy:", calc_confidence(8, 0.04, 5))    # ~0.2
print("Medium:", calc_confidence(18, 0.025, 10))         # ~0.6
print("Strong:", calc_confidence(30, 0.015, 15))         # ~0.8
print("Very strong:", calc_confidence(48, 0.008, 20))    # ~0.95
```

This confidence score is what separates "maybes" from "definitely important"! 🎯

# scrollcast v0.2 check-in (T+2 weeks)
**Date:** 2026-05-14

---

## Quick recap — what shipped at v0.1

- **Skill scaffolds a full Vite + React + MDX → single `.html` pipeline**, with Apple-design-language baseline, CSS scroll-snap + Motion (Framer Motion) hybrid keyboard/free-scroll, six section components (Hero, Problem, Solution, Proof, Interactive, CTA), and a `--brand` flag backed by CSS-variable brand-preset files.
- **One widget fully working** (`ROICalculator`); five stubbed in `widgets/registry.ts` and not yet implemented: `ComparisonSlider`, `MetricTicker`, `SimpleChart`, `ThresholdDemo`, `FeatureToggle`. Playwright `--record` MP4 pipeline and per-property brand presets (EdgeProof / Vigil / AxisX) are also on the v0.2 backlog.
- **Your post-demo note (2026-04-30):** design and flow landed well, but you want *richer motion graphics* in v0.2 — parallax, animated backgrounds, magicui / aceternity showpieces as options.

---

## Local follow-up prompt

Paste this into a new local Claude Code session to pick up v0.2 planning:

```
Hey — this is a scrollcast v0.2 planning session, T+2 weeks after v0.1 shipped.

First, orient yourself:

1. Run: ls ~/Desktop | grep -i scroll
   (find any scrollcast projects I've created since launch)

2. Run: ls -la ~/.claude/skills/scrollcast/template/src/widgets/
   and:  ls -la ~/.claude/skills/scrollcast/template/src/sections/
   (check mtimes to see what I've modified since 2026-04-30)

Then ask me these five questions — one at a time or all at once, your call:

a) Which of the five stubbed widgets did you actually reach for and find missing?
   (ComparisonSlider, MetricTicker, SimpleChart, ThresholdDemo, FeatureToggle)

b) Which of those five have you had zero need for so far?

c) Did the Playwright `--record` MP4 pipeline come up in any real usage?

d) Have any per-property brand needs surfaced — EdgeProof, Vigil, AxisX, or others?

e) How has motion-graphics richness played out in practice? Is the current
   Motion + scroll-snap baseline enough, or is it time to bring in magicui /
   aceternity showpieces, a GSAP escape hatch, or parallax backgrounds for v0.2?

Based on my answers, propose a v0.2 implementation order:
- Top 2–3 widgets to implement first
- Whether `--record` lands in this sprint or slips
- Whether motion-graphics richness moves to the top of the list or stays a
  nice-to-have

Then offer to scaffold the v0.2 work immediately.

Special case: if I haven't used scrollcast at all yet, ask whether to archive
it, keep it dormant, or refine based on what blocked me from reaching for it.
```

---

## Direct questions for Matthew

In case you'd rather answer here in the routines UI:

1. Which of the five stub widgets did you actually reach for and find missing? (`ComparisonSlider`, `MetricTicker`, `SimpleChart`, `ThresholdDemo`, `FeatureToggle`)
2. Which of those five have you had zero need for — safe to deprioritize?
3. Did the Playwright `--record` MP4 export come up in any real usage?
4. Have any per-property brand needs surfaced (EdgeProof / Vigil / AxisX / other)?
5. How is the motion-graphics baseline holding up — is Motion + scroll-snap enough, or is it time to slot in magicui / aceternity showpieces or a GSAP escape hatch for v0.2?

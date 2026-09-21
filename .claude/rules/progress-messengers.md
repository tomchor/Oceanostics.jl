---
paths:
  - "src/ProgressMessengers/**"
  - "test/test_progress_messengers.jl"
  - "docs/src/progress_messengers.md"
---
# Progress messengers

A new leaf messenger wraps its formatted number (the result of `@sprintf` or `prettytime`) in `ColoredNumber`, so the value takes the configurable `NUMBER_CRAYON` color; prefix and unit text stay plain `String`.
`ColoredNumber` concatenates to a `String` with the color codes included, so `+` (comma-separated) and `*` (concatenation) compose colored and plain messengers alike.

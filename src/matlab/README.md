# Mixing Scripts For Speech Enhancement Project

1. The training/testing dataset now are split accoring to total time of each group. This makes more sense as portions are more stable as examples are randomly shuffled.

2. Split information written to log.json of training/testing folder.

3. Add inference function.

4. feature(s, label, flag): generate spectrum data to /root/training/spectrum/s_*.mat   
# Matched-margin control — predictions written BEFORE the run (post-hoc experiment, designed after seeing the pilot30 results)
Question: is the success loss of Adaptive+ObsACI (48.7 % success, 59.3 % finished, 7.3 % collisions, mean speed 11.6 m/s; pooled shift, 30 seeds) caused by the SIZE of its obstacle margin, or by coverage/adaptation as such?
Controllers: Adaptive+ObsStaticCP with the static margin multiplied by k = 1.5 and k = 2.0 (capped at OBS_QCAP = 3 m, like the ACI margin). Reference numbers from the pilot30 run: ObsStaticCP (k = 1): 62.0 % success, 75.3 % finished, 8.7 % collisions; ObsACI: see above.
- M1 (size drives the cost): k = 2.0 gives success <= 55 % AND finished <= 70 % (i.e. close to ObsACI). If k = 2.0 stays >= 58 % success and >= 72 % finished, M1 is REJECTED (the cost comes from the scenario-dependent / adaptive shape of the ACI margin, not its size).
- M2 (collision reduction does not need high coverage): collisions at k = 1.5 and k = 2.0 are <= 10 %.
Both are evaluated on pooled shift scenarios (5), 30 seeds, same seeds as the main run. Nothing else is looked at to decide M1/M2.

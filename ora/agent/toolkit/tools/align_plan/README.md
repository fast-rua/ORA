# align_plan

Derive the current target centroid from its public mask, depth and calibration,
then synthesize the released named approach directions and jaw rolls. Only the
planner class supplies IK results. Missing or changed replay queries fail
NotRecorded; images and geometry are recorded/pure computation. No motion.

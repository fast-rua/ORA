// Names the stages share: their ids, the notices one stage sends another, and
// the schema of a projection transition record.

export const MATERIALIZE = "materialize";
export const STRIDE_AGING = "stride_aging";
export const EAGER_EXTERNALIZE = "eager_externalize";
export const MEDIA_WINDOW = "media_window";
export const PRESSURE_EXTERNALIZE = "pressure_externalize";
export const CAMERA_DELIVERY = "camera_delivery";
export const EPISODE_CAPSULE = "episode_capsule";
export const EMERGENCY_CHECKPOINT = "emergency_checkpoint";
export const OVER_LIMIT_GUARD = "over_limit_guard";
export const EXCHANGE_AGING = "exchange_aging";
export const REASONING_REPLAY = "reasoning_replay";
export const HARD_BACKSTOP = "hard_backstop";

/** episode_capsule to emergency_checkpoint: a new capsule covers the active checkpoint. */
export const SUPERSEDED_BY_EPISODE_CAPSULE = "superseded_by_episode_capsule";
/** exchange_aging to reasoning_replay: an aging moved the aged frontier (detail `aged`, `exchanges`). */
export const EXCHANGES_AGED = "exchanges_aged";
/** hard_backstop to exchange_aging and reasoning_replay: age all but the backstop's tail (detail `target`, `observation_groups`). */
export const BACKSTOP_AGED = "backstop_aged";

export const TRANSITION_SCHEMA = "ora.context-projection-transition.v4";

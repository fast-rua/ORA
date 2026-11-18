"""One colour table, and the ceiling on grounding candidates derived from it.

Two three-entry tuples used to sit inline in the two rendering paths and
disagree about what happens past the third candidate: the overview zipped
non-strictly and drew nothing, while the enlarged preview indexed modulo
three and painted the fourth in the first one's colour. Neither raised, so
how much a policy could be shown was decided by the length of a tuple
nobody read as a bound.
"""

from __future__ import annotations

from ora.agent.toolkit.lib._vision_palette import (
    GROUNDING_CANDIDATE_COLOURS, GROUNDING_MAX_CANDIDATES, GROUNDING_PREVIEW_RENDER_LIMIT, grounding_candidate_colour,
)

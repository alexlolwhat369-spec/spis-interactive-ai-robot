# Object Guessing Algorithm

## Acceleration Brief

- **GitHub gate:** adapt.
- **Candidate:** [Aaklon/akinator](https://github.com/Aaklon/akinator) documents expected information gain for a fast Akinator-like engine, but it is Go and AGPL-3.0.
- **Decision:** build a small local Python implementation from the mathematical idea only. No external code, dependency, web API, or user profile is used.
- **Why:** the robot must work without a paid service, be explainable during a science fair, and later run on Raspberry Pi.

## Algorithm

Each object starts with equal probability. For each unasked attribute, the game calculates the expected reduction in Shannon entropy over five possible answers:

`yes`, `probably`, `maybe`, `probably not`, `no`.

It asks the attribute with the highest expected information gain, then updates every object score with Bayes-style likelihoods. A definite answer has more weight than `probably`; `maybe` leaves the distribution unchanged. The game guesses only when the first candidate is both confident and clearly ahead of the second candidate. After 12 questions without enough confidence, it asks the visitor for the intended object instead of making a weak guess.

This follows the general information-gain idea described in the external project and in the [Akinator question-selection analysis](https://www.mdpi.com/2504-2289/7/1/26). The original Akinator algorithm itself is not public, so this is our own transparent object-game implementation.

## Category-First Questions

The catalog assigns each current object one primary category: `technology`,
`school`, `food_drink`, or `play_mobility`. The game first selects the most
informative remaining category question. When one category holds at least 65%
of the current probability, it favors questions from that category. A small
category bonus keeps the conversation coherent, while expected information
gain can still select a much stronger general question when it would separate
the remaining candidates better. If the visitor says no to a category, its
probability falls and the next category question is selected from the remaining
candidates.

Every catalog question must also pass a conversation-quality check: it should
separate plausible remaining objects and tell the visitor something meaningful.
The catalog deliberately avoids asking attributes that are usually implied by
an accepted category, such as asking about a battery immediately after the
visitor has already confirmed that an item is electronic.

## Catalog Rule

Every object must have a unique set of attributes. Otherwise, no question strategy can distinguish two identical candidates. `tests/test_object_game.py` checks this rule and simulates truthful answers for every catalog object.

Questions normally test one attribute, such as "Is it electronic?" In that
case, `no` means that the object does not have the attribute. A comparison can
also explicitly define both sides, such as `portable` versus `stationary`.
The engine validates that every catalog object belongs to exactly one side
before such a comparison is allowed. This prevents an ambiguous question from
treating "no" as proof of a category the visitor did not actually choose.

## Learning After a Miss

The correction flow is adapted from the useful idea in
[GabrielTerra55/Alkinator_EDD](https://github.com/GabrielTerra55/Alkinator_EDD):
after a wrong guess, ask for the intended object and one question that separates
it from the wrong guess. That source is MIT licensed, but this project uses a
new implementation rather than its desktop binary-tree code.

Our robot writes the correction to `data/pending_object_suggestions.jsonl` for
a teacher or project owner to review. It does not quietly add unverified facts
to `object_catalog.json`. The probabilistic engine then remains free to choose
the next best question across the full catalog instead of being locked into one
binary-tree path.

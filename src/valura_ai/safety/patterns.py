"""Safety patterns and per-category professional refusals.

Layered detection:

  Layer 1  — Category phrases    : substrings naming a harmful concept
                                   ('insider trading', 'pump and dump', 'launder cash').
                                   Weight 1.
  Layer 2  — Intent phrases      : actor + action combinations that imply
                                   execution ('CFO friend hinted', 'help me launder',
                                   'trade on this rumor'). Weight 2.
  Layer 3  — Methodology markers : how-to framings shared across categories
                                   ('how do I', 'how would someone', 'best way to',
                                   'common methods', 'and how it's done').
  Layer 4  — Evasion markers     : hypothetical and concealment cues that
                                   signal attempt to extract harmful info under
                                   educational cover ('just curious', 'theoretically',
                                   'asking for a friend', 'between you and me',
                                   'won't tell anyone', "how would anyone know",
                                   'without getting caught').
  Layer 5  — Educational markers : defensive framings that legitimately reduce
                                   the score for pure phrase matches ('what is',
                                   'explain the difference', 'how was the X case',
                                   regulatory references, named historical cases).

The decision logic that consumes these layers lives in ``guard.py``.
This module is data-only by design — patterns can be reviewed and edited
without touching code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryPatterns:
    name: str
    # Plain substrings (lowercased) — fast `in` test. Weight 1 each.
    phrases: tuple[str, ...]
    # Plain substrings — actor + action combos. Weight 2 each.
    intent_phrases: tuple[str, ...]
    # Compiled regex patterns — for combinations substring can't capture
    # cleanly (e.g. "we have a [trial|launch] coming"). Each match weight 2.
    intent_regex: tuple[re.Pattern[str], ...]
    # Distinct, professional refusal — per the assignment, no generic message.
    refusal: str


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

INSIDER_TRADING = CategoryPatterns(
    name="insider_trading",
    phrases=(
        "insider trading",
        "inside tip",
        "inside info",
        "inside information",
        "material non-public",
        "material nonpublic",
        "non-public information",
        "nonpublic information",
        "mnpi",
        "tipped off",
        "before earnings",
        "before the earnings",
        "before the announcement",
        "before the merger",
        "before the deal",
        "deal team",
        "c-suite",
        "csuite",
        "trial result",
    ),
    intent_phrases=(
        "buddy at",
        "friend at apple",
        "friend at google",
        "friend at microsoft",
        "ceo told me",
        "cfo told me",
        "cto told me",
        "ceo friend",
        "cfo friend",
        "cto friend",
        "from someone in the c-suite",
        "trade on this rumor",
        "trade on this tip",
        "trade on it",
        "act on that",
        "act on this",
        "have inside info",
        "have an inside",
        "got an inside",
        "got inside",
        "leaked document",
        "use information from a leaked",
        "from a leaked",
        "happened to learn something material",
        "they tip me",
        "ok to trade on it",
        "is it ok to trade",
        "i trade",  # appears with "they tip me, I trade"
    ),
    intent_regex=(
        # "we have a/an [material event] coming/announcement/next week"
        re.compile(
            r"\bwe (?:have|got)(?:\s+(?:a|an))?\s+"
            r"(?:trial|earnings|launch|merger|deal|announcement|guidance|product launch|restructuring)"
        ),
        # "I work at <X> and <... material event ...>"
        re.compile(
            r"\bi\s+work\s+at\s+\w+.*\b"
            r"(?:trial|earnings|launch|merger|deal|announcement|guidance|results)\b"
        ),
        # "[someone at <Co>] told me/hinted [material outcome]"
        re.compile(
            r"\b(?:buddy|friend|cousin|partner|spouse|brother|sister|colleague|coworker)\s+at\s+\w+.*"
            r"\b(?:told me|hinted|tipped|whispered)\b"
        ),
        # "should I (short|trade|buy calls|buy puts)" combined with insider context phrase
        re.compile(
            r"\b(?:should i|can i)\s+(?:short|buy calls|buy puts|trade)\b"
        ),
    ),
    refusal=(
        "I can't help with trading on material non-public information. "
        "Doing so violates U.S. securities laws and similar regulations in most "
        "jurisdictions, and exposes you to civil and criminal liability. "
        "If it would help, I can explain how insider-trading rules work, walk through "
        "a notable enforcement case, or discuss publicly available information about "
        "the company you mentioned."
    ),
)

MARKET_MANIPULATION = CategoryPatterns(
    name="market_manipulation",
    phrases=(
        "pump and dump",
        "pump-and-dump",
        "wash trade",
        "wash trading",
        "spoofing",
        "spoof orders",
        "spoof the",
        "paint the tape",
        "marking the close",
        "mark the close",
        "mark up nav",
        "fake volume",
        "layering orders",
    ),
    intent_phrases=(
        "help me organize a pump",
        "i want to wash trade",
        "wash trade with my own",
        "coordinate buying with",
        "coordinated buying",
        "discord group to spike",
        "discord pump",
        "telegram pump",
        "to spike this stock",
        "to fake volume",
    ),
    intent_regex=(
        # "how do I/would someone [spoof|wash trade|paint|pump|coordinate]"
        re.compile(
            r"\bhow (?:do|would|can|should) (?:i|you|someone|people)\b.*"
            r"\b(?:spoof|wash trade|paint the tape|pump|coordinate buying|fake volume)\b"
        ),
    ),
    refusal=(
        "I can't help with strategies that distort market prices. "
        "Pump-and-dump schemes, spoofing, wash trading, coordinated price action, "
        "and similar conduct are prohibited under exchange rules and securities laws. "
        "I'm happy to explain how regulators detect manipulation, walk through a "
        "historical case, or discuss what protections exist for retail investors."
    ),
)

MONEY_LAUNDERING = CategoryPatterns(
    name="money_laundering",
    phrases=(
        "launder cash",
        "launder money",
        "money laundering",
        "clean money",
        "clean cash",
        "structure deposits",
        "structuring deposits",
        "structure cash deposits",
        "layer transactions",
        "obscure the source",
        "fly under the radar",
        "under the radar",
        "avoid 10k reporting",
        "avoid the 10k",
        "avoid reporting",
    ),
    intent_phrases=(
        "i need to clean",
        "i need to launder",
        "best way to launder",
        "show me how to layer",
        "what's the process for laundering",
        "process for laundering",
        "launder through",
        "launder it through",
    ),
    intent_regex=(
        # "how do I/people structure deposits"
        re.compile(
            r"\bhow (?:do|would|can|should) (?:i|you|someone|people)\b.*"
            r"\b(?:structure|launder|clean|layer)\b"
        ),
    ),
    refusal=(
        "I can't help with concealing the source of funds or evading "
        "transaction-reporting requirements. These activities violate "
        "anti-money-laundering laws (the U.S. Bank Secrecy Act, the EU AMLD, and "
        "equivalent regimes elsewhere). If you have a legitimate compliance "
        "question — how AML monitoring works, what reporting thresholds apply, or "
        "how legitimate funds should be transferred — I'm happy to discuss."
    ),
)

GUARANTEED_RETURNS = CategoryPatterns(
    name="guaranteed_returns",
    phrases=(
        "guarantee 50%",
        "guarantee 20%",
        "guarantee 100%",
        "guaranteed return",
        "guaranteed returns",
        "guaranteed 50",
        "guaranteed 20",
        "guaranteed profit",
        "promise me a",
        "sure thing",
        "won't lose money",
        "wont lose money",
        "no risk play",
        "no-risk play",
        "double my money",
        "double my account",
    ),
    intent_phrases=(
        "give me a stock that will guarantee",
        "guarantee me",
        "what trade gives a guaranteed",
        "what trade gives me a guaranteed",
        "i want a sure thing",
        "what won't lose money",
        "promise me",
    ),
    intent_regex=(),
    refusal=(
        "No legitimate investment can guarantee returns or eliminate the risk of "
        "loss — anyone telling you otherwise is either misinformed or running a "
        "scam. I won't suggest trades framed as guarantees. I can discuss historical "
        "returns, how to think about risk-adjusted return, or the difference between "
        "low-risk and risk-free if that helps."
    ),
)

RECKLESS_ADVICE = CategoryPatterns(
    name="reckless_advice",
    phrases=(
        "emergency fund into",
        "emergency fund on",
        "yolo into",
        "0dte options",
        "0dte calls",
        "mortgage my house to",
        "mortgage the house to",
        "kid's college fund",
        "kids college fund",
        "college fund on leveraged",
        "college fund into leveraged",
        "shitcoins",
        "shit coins",
        "heloc and yolo",
        "heloc to yolo",
        "all in on a single",
    ),
    intent_phrases=(
        "tell me to mortgage",
        "tell me to put my entire",
        "use my kid's college",
        "use my kids college",
        "take a heloc and yolo",
        "buy more nvda calls",
        "go all in on options",
        "should i put my entire emergency",
    ),
    intent_regex=(
        # "use my [savings_term] (to|on|for) [speculative]"
        re.compile(
            r"\b(?:use|put|throw|dump)\s+(?:my\s+)?"
            r"(?:emergency fund|college fund|retirement|savings|rent|mortgage)\s+"
            r"(?:into|on|for)\s+(?:options|calls|puts|leveraged|shitcoins|crypto|0dte)"
        ),
    ),
    refusal=(
        "I won't endorse putting essential funds — emergency reserves, college "
        "savings, mortgaged equity — into highly speculative or leveraged "
        "positions. The downside scenario is severe and the math rarely favors "
        "the investor. I'm glad to walk through position sizing, risk-management "
        "frameworks, or how to think about leverage in a balanced portfolio."
    ),
)

FRAUD = CategoryPatterns(
    name="fraud",
    phrases=(
        "boiler room",
        "fake earnings",
        "fake earnings press",
        "fake press release",
        "fabricate earnings",
        "fabricate disclosures",
        "run a ponzi",
        "set up a ponzi",
        "set up a boiler",
        "without getting caught",  # generic enough to be evasion, but in fraud context blocks
    ),
    intent_phrases=(
        "help me write a fake",
        "how do i run a ponzi",
        "how do i set up a boiler",
        "best way to set up a boiler",
        "best way to run a ponzi",
    ),
    intent_regex=(),
    refusal=(
        "I can't help with fabricating, misrepresenting, or executing fraudulent "
        "investment schemes — this includes fake disclosures, Ponzi structures, and "
        "high-pressure sales operations. These are criminal offenses. "
        "If you're trying to recognise or avoid fraud, I can explain the warning "
        "signs and walk through the structure of well-known cases."
    ),
)

TAX_EVASION = CategoryPatterns(
    name="tax_evasion",
    phrases=(
        "hide gains",
        "hide income",
        "hide from the irs",
        "hide it from the irs",
        "dodge taxes",
        "dodge tax",
        "offshore setup",
        "never report this income",
        "shuffle assets to dodge",
        "didn't report it",
        "didnt report it",
        "wont report it",
        "won't report it",
        "evade taxes",
        "evading taxes",
    ),
    intent_phrases=(
        "how do i hide",
        "best offshore setup",
        "show me how to shuffle",
        "how do i dodge",
        "easiest way to dodge",
    ),
    intent_regex=(
        # "[friend/I/we] want(s) to hide [income/gains/money] from [taxes/irs]"
        re.compile(
            r"\b(?:hide|conceal|stash)\s+(?:my|the|some|all)?\s*"
            r"(?:income|gains|money|profit|profits)\s+from\s+(?:the\s+)?(?:irs|hmrc|tax|taxes)"
        ),
    ),
    refusal=(
        "I can't help with concealing income, hiding gains, or constructing "
        "arrangements designed to avoid lawful tax obligations. Tax evasion is a "
        "criminal offense in most jurisdictions. I'm happy to discuss the "
        "difference between lawful tax planning and tax evasion, or specific legal "
        "strategies — tax-loss harvesting, account selection, deferral — if useful."
    ),
)

ALL_CATEGORIES: tuple[CategoryPatterns, ...] = (
    INSIDER_TRADING,
    MARKET_MANIPULATION,
    MONEY_LAUNDERING,
    GUARANTEED_RETURNS,
    RECKLESS_ADVICE,
    FRAUD,
    TAX_EVASION,
)


# ---------------------------------------------------------------------------
# Cross-category markers
# ---------------------------------------------------------------------------

# Methodology framings: "how do I do X", "best way to do X", etc. When paired
# with any category signal we treat the request as a methodology ask (block),
# regardless of educational framing in the rest of the sentence.
METHODOLOGY_MARKERS: tuple[str, ...] = (
    "how do i ",
    "how can i ",
    "how would i ",
    "how do you ",
    "how would you ",
    "how would someone",
    "how do people",
    "how does someone",
    "how do they",
    "show me how to",
    "what's the process for",
    "what is the process for",
    "best way to",
    "easiest way to",
    "step by step",
    "step-by-step",
    "common methods",
    "and how it's done",
    "and how its done",
    "and how it is done",
    "and how to do it",
    "what are the methods",
    "what are common methods",
)

# Evasion / concealment cues — "I'm hiding the ask behind innocent framing".
# When paired with any category signal we treat as a block.
EVASION_MARKERS: tuple[str, ...] = (
    "just curious",
    "purely hypothetical",
    "theoretically",
    "out of curiosity",
    "asking for a friend",
    "for a friend",
    "between you and me",
    "won't tell anyone",
    "wont tell anyone",
    "wouldn't tell anyone",
    "off the record",
    "how would anyone know",
    "without getting caught",
    "fly under the radar",
    "no one will know",
    "nobody will know",
    "i won't tell",
    "i wont tell",
)

# Educational framings — when present, a SINGLE phrase hit (no methodology, no
# evasion, no intent) is treated as a defensive question rather than harmful.
EDUCATIONAL_MARKERS: tuple[str, ...] = (
    "what is ",
    "what's ",
    "what are ",
    "what does ",
    "what was ",
    "explain ",
    "explain why",
    "explain how",
    "is it true",
    "are there any legal",
    "are there legal",
    "what's the difference",
    "what is the difference",
    "case study",
    "historical",
    "how did the",
    "how was",
    "regulatory",
    "regulations",
    "compliance",
    "consumer protection",
    "red flag",
    "is a red flag",
    "are usually a red flag",
    "10b5-1",
    "10b5",
    "bsa",
    "aml",
    "is it considered",
    "are inflation-linked",
    "considered risk-free",
)

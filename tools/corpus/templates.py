"""Hand-authored call scenarios with ground-truth labels.

Each ``Scenario`` is a dialogue skeleton plus the labels a perfect system would
produce for it. Placeholders in the turn text are filled at generation time:

    ``{CARD}``      new value for that PII type
    ``{CARD@1}``    reuse slot 1 of that type (so a repeated card matches)
    ``{CARD@1/1}``  first half of slot 1's value  -- split across turns
    ``{CARD@1/2}``  second half of slot 1's value

Splitting a value across two turns is not decoration: it is the single hardest
case for a streaming redactor, because neither segment on its own contains a
recognisable card. The scenarios that use it are the reason the redactor has a
cross-segment carry buffer at all.

Labels are exact by construction rather than annotated after the fact. That is
a strength for PII spans (no annotator drift) and a weakness for the
judgement labels -- intent, sentiment, resolution -- where the author's
intent may not match what a careful human rater would say. EVAL.md is explicit
about which numbers inherit that bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Turn = tuple[str, str]  # (speaker, text)

INTENTS = (
    "billing_dispute",
    "payment_arrangement",
    "account_closure",
    "technical_support",
    "fraud_report",
    "address_change",
    "refund_request",
    "plan_upgrade",
    "balance_inquiry",
    "collections",
    "card_replacement",
    "password_reset",
    "general_inquiry",
)


@dataclass(frozen=True)
class Labels:
    primary_intent: str
    resolution: str  # resolved | unresolved | escalated | follow_up_required
    sentiment_start: float
    sentiment_end: float
    escalation_risk: str  # none | low | medium | high
    escalated: bool = False
    secondary_intents: tuple[str, ...] = ()
    required_disclosures: tuple[str, ...] = ()
    disclosures_given: tuple[str, ...] = ()
    compliance_violations: tuple[str, ...] = ()
    action_items: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class Scenario:
    slug: str
    category: str  # clean | pii_heavy | compliance | escalation | ambiguous | adversarial
    turns: tuple[Turn, ...]
    labels: Labels
    agent_name: str = "Ray"
    #: Surface forms to prefer for numeric PII in this scenario. Adversarial
    #: scenarios pin these so the hard cases are guaranteed present rather than
    #: left to the RNG.
    surface_forms: tuple[str, ...] = field(default=("plain", "spaced", "hyphenated"))


GREETING_COMPLIANT = (
    "agent",
    "Thanks for calling Northwind Financial, my name is {AGENT}. "
    "Just so you know, this call is recorded for quality and training. How can I help?",
)
GREETING_BARE = ("agent", "Northwind Financial, this is {AGENT}. What can I do for you?")


SCENARIOS: tuple[Scenario, ...] = (
    # ---------------------------------------------------------------- clean
    Scenario(
        slug="clean_balance_inquiry",
        category="clean",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "Hi, I just wanted to check the balance on my checking account."),
            ("agent", "Happy to help with that. Can you confirm the date of birth on the account?"),
            ("customer", "Sure, it's {DOB}."),
            ("agent", "Thank you, that matches. Your available balance is one thousand four hundred and six dollars."),
            ("customer", "Great, that's what I expected. Thanks."),
            ("agent", "Anything else I can take care of today?"),
            ("customer", "No, that's all. Have a good one."),
            ("agent", "You too. Thanks for calling Northwind."),
        ),
        labels=Labels(
            primary_intent="balance_inquiry",
            resolution="resolved",
            sentiment_start=0.2,
            sentiment_end=0.6,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=(),
        ),
    ),
    Scenario(
        slug="clean_password_reset",
        category="clean",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I'm locked out of online banking. It keeps saying my password is wrong."),
            ("agent", "I can reset that. First, let me verify your identity — can you confirm the last four of the card on file?"),
            ("customer", "Yes, it ends in four four four four."),
            ("agent", "Perfect. I've sent a one-time code to the phone ending in one four two. Can you read me the code?"),
            ("customer", "It's eight one nine three."),
            ("agent", "Got it. Your password is reset — you'll be prompted to choose a new one at next login."),
            ("customer", "That worked. Thank you."),
            ("agent", "My pleasure. Anything else?"),
            ("customer", "Nope, all set."),
        ),
        labels=Labels(
            primary_intent="password_reset",
            resolution="resolved",
            sentiment_start=-0.2,
            sentiment_end=0.5,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
        ),
    ),
    Scenario(
        slug="clean_technical_support",
        category="clean",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "The mobile app crashes every time I open the statements tab."),
            ("agent", "Sorry about that. Which phone are you on, and what app version?"),
            ("customer", "An iPhone, and the version says six point two point one."),
            ("agent", "That's a known issue on six point two point one. Version six point two point two went out this morning and fixes it."),
            ("customer", "Let me update. Okay, it's installing."),
            ("agent", "While that runs — statements are also available on the web portal if you need one right now."),
            ("customer", "It opened fine after the update. Thanks for the quick answer."),
            ("agent", "Glad that sorted it. Have a good day."),
        ),
        labels=Labels(
            primary_intent="technical_support",
            resolution="resolved",
            sentiment_start=-0.3,
            sentiment_end=0.5,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
        ),
    ),
    Scenario(
        slug="clean_general_inquiry_hours",
        category="clean",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "Quick question — what time does the Fenwick Road branch close on Saturdays?"),
            ("agent", "That branch closes at two in the afternoon on Saturdays, and it's closed Sundays."),
            ("customer", "Perfect, that's all I needed."),
            ("agent", "Easy one. Thanks for calling."),
        ),
        labels=Labels(
            primary_intent="general_inquiry",
            resolution="resolved",
            sentiment_start=0.3,
            sentiment_end=0.5,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
        ),
    ),
    # ------------------------------------------------------------- pii_heavy
    Scenario(
        slug="pii_card_replacement",
        category="pii_heavy",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "My card was damaged, I need a replacement. My name is {PERSON}."),
            ("agent", "I can order that. Before I make changes, I need to verify your identity — can you confirm your date of birth?"),
            ("customer", "It's {DOB}."),
            ("agent", "Thank you. And the best number to reach you?"),
            ("customer", "{PHONE}. You can also email me at {EMAIL}."),
            ("agent", "Noted. Is the card still going to {ADDRESS}?"),
            ("customer", "Yes, same address."),
            ("agent", "Your replacement card ending in the same last four ships today, arriving in five to seven business days."),
            ("customer", "Appreciate it, thanks."),
            ("agent", "Of course. Anything else?"),
            ("customer", "That's everything."),
        ),
        labels=Labels(
            primary_intent="card_replacement",
            resolution="resolved",
            sentiment_start=0.0,
            sentiment_end=0.4,
            escalation_risk="none",
            required_disclosures=("REC-001", "VERIF-004"),
            disclosures_given=("REC-001", "VERIF-004"),
            action_items=("Ship replacement card to address on file",),
        ),
    ),
    Scenario(
        slug="pii_fraud_report_full_profile",
        category="pii_heavy",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "There are charges on my account I didn't make. This is {PERSON}."),
            ("agent", "I'm sorry, let's get that locked down. Can you confirm your date of birth and the account number?"),
            ("customer", "Date of birth {DOB}, account {ACCOUNT}."),
            ("agent", "Thank you. I see three transactions flagged. Was the card in your possession?"),
            ("customer", "Yes, it's right here. The card is {CARD}."),
            ("agent", "I have the last four, that's all I need. I'm freezing the card now."),
            ("customer", "Also my social is {SSN} if you need it for the report."),
            ("agent", "I don't need that for this. The fraud claim is filed and provisional credit posts in two business days."),
            ("customer", "Okay. Can you email me the confirmation at {EMAIL}?"),
            ("agent", "Sent. You'll also get a text at {PHONE}."),
            ("customer", "Thank you for handling that quickly."),
        ),
        labels=Labels(
            primary_intent="fraud_report",
            resolution="follow_up_required",
            sentiment_start=-0.6,
            sentiment_end=0.2,
            escalation_risk="low",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=(
                "File fraud claim and freeze card",
                "Post provisional credit within two business days",
                "Email claim confirmation to customer",
            ),
            notes="Customer volunteers SSN unprompted; agent correctly declines to use it.",
        ),
    ),
    Scenario(
        slug="pii_address_change_verified",
        category="pii_heavy",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I moved and need to update my address. It's {PERSON}, account {ACCOUNT}."),
            ("agent", "Let me verify your identity first — what's the date of birth on the account?"),
            ("customer", "{DOB}."),
            ("agent", "Thanks. What's the new address?"),
            ("customer", "{ADDRESS}. And my new number is {PHONE}."),
            ("agent", "Updated both. Statements will go to the new address starting next cycle."),
            ("customer", "Can you also confirm my email on file? Should be {EMAIL}."),
            ("agent", "That's what I have. All set."),
            ("customer", "Great, thanks."),
        ),
        labels=Labels(
            primary_intent="address_change",
            resolution="resolved",
            sentiment_start=0.1,
            sentiment_end=0.4,
            escalation_risk="none",
            required_disclosures=("REC-001", "VERIF-004"),
            disclosures_given=("REC-001", "VERIF-004"),
            action_items=("Update mailing address and phone on file",),
        ),
    ),
    Scenario(
        slug="pii_payment_arrangement",
        category="pii_heavy",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I want to set up a payment plan. I can't pay the full amount this month."),
            ("agent", "We can do that. Can I get the account number?"),
            ("customer", "{ACCOUNT}, and my date of birth is {DOB}."),
            ("agent", "Thank you. What amount can you manage monthly?"),
            ("customer", "About two hundred a month. Use the card ending in the last four of {CARD}."),
            ("agent", "I'll take just the last four. Payment plan is three payments of two hundred and eleven dollars."),
            ("customer", "That works. Send confirmation to {EMAIL}."),
            ("agent", "Done. First payment draws on the fifteenth."),
            ("customer", "Thanks, that's a relief."),
        ),
        labels=Labels(
            primary_intent="payment_arrangement",
            resolution="resolved",
            sentiment_start=-0.4,
            sentiment_end=0.5,
            escalation_risk="low",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=(
                "Set up three-payment arrangement",
                "Email payment schedule confirmation",
            ),
        ),
    ),
    Scenario(
        slug="pii_refund_bank_details",
        category="pii_heavy",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I was double charged for the annual fee and want it refunded."),
            ("agent", "Let me look. Can you confirm the account?"),
            ("customer", "Account {ACCOUNT}, name {PERSON}, date of birth {DOB}."),
            ("agent", "I see two ninety-five dollar charges on the same day. That's clearly duplicated."),
            ("customer", "Right. Refund it to the card {CARD}."),
            ("agent", "It goes back to the original card automatically, I only need the last four."),
            ("customer", "Fine. My phone is {PHONE} if you need to reach me."),
            ("agent", "Refund submitted, five to seven business days."),
            ("customer", "Okay, thank you."),
        ),
        labels=Labels(
            primary_intent="refund_request",
            resolution="follow_up_required",
            sentiment_start=-0.4,
            sentiment_end=0.3,
            escalation_risk="low",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=("Submit duplicate-charge refund of ninety-five dollars",),
        ),
    ),
    # ------------------------------------------------------------ compliance
    Scenario(
        slug="compliance_missing_recording_disclosure",
        category="compliance",
        turns=(
            GREETING_BARE,
            ("customer", "I need to dispute a charge from last Tuesday."),
            ("agent", "Sure. What's the account number?"),
            ("customer", "{ACCOUNT}."),
            ("agent", "And your date of birth?"),
            ("customer", "{DOB}."),
            ("agent", "Thanks. I see the charge — forty-two dollars at a gas station."),
            ("customer", "I wasn't in that state last Tuesday."),
            ("agent", "I'll open a dispute. You'll hear back in ten business days."),
            ("customer", "Alright."),
        ),
        labels=Labels(
            primary_intent="billing_dispute",
            resolution="follow_up_required",
            sentiment_start=-0.2,
            sentiment_end=0.0,
            escalation_risk="low",
            required_disclosures=("REC-001",),
            disclosures_given=(),
            compliance_violations=("REC-001",),
            action_items=("Open dispute for gas station charge",),
            notes="Agent collects DOB and account number with no recording disclosure at all.",
        ),
    ),
    Scenario(
        slug="compliance_pci_full_readback",
        category="compliance",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I want to make a payment over the phone."),
            ("agent", "Happy to take that. Go ahead and read me the full card number."),
            ("customer", "It's {CARD}."),
            ("agent", "Let me read that back to me to confirm — {CARD@1}."),
            ("customer", "That's right."),
            ("agent", "Payment of three hundred dollars is authorised."),
            ("customer", "Thanks."),
        ),
        labels=Labels(
            primary_intent="payment_arrangement",
            resolution="resolved",
            sentiment_start=0.1,
            sentiment_end=0.3,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            compliance_violations=("PCI-002",),
            notes="Agent both solicits and repeats a full PAN. Two distinct PCI-002 instances.",
        ),
    ),
    Scenario(
        slug="compliance_missing_mini_miranda",
        category="compliance",
        turns=(
            ("agent", "This is {AGENT} calling from Northwind Recovery regarding your past due balance. This call is recorded."),
            ("customer", "Okay, what's this about?"),
            ("agent", "You have a balance of one thousand two hundred dollars that's ninety days past due."),
            ("customer", "I know, I've been out of work."),
            ("agent", "We need to arrange payment. Can you pay half today?"),
            ("customer", "I really can't right now."),
            ("agent", "What can you pay?"),
            ("customer", "Maybe fifty dollars."),
            ("agent", "I'll note that and we'll follow up next month."),
        ),
        labels=Labels(
            primary_intent="collections",
            resolution="follow_up_required",
            sentiment_start=-0.3,
            sentiment_end=-0.5,
            escalation_risk="medium",
            required_disclosures=("REC-001", "MINI-003"),
            disclosures_given=("REC-001",),
            compliance_violations=("MINI-003",),
            action_items=("Follow up on fifty dollar payment next month",),
            notes="Recording disclosure given, mini-Miranda absent on a collections call.",
        ),
    ),
    Scenario(
        slug="compliance_prohibited_guarantee",
        category="compliance",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I filed a dispute two weeks ago and haven't heard anything."),
            ("agent", "Let me check. It's still under review with the merchant."),
            ("customer", "Am I going to get my money back?"),
            ("agent", "I guarantee you'll get your money back, don't worry about it."),
            ("customer", "Okay, that's reassuring."),
            ("agent", "You'll see the credit soon."),
            ("customer", "Thanks."),
        ),
        labels=Labels(
            primary_intent="billing_dispute",
            resolution="unresolved",
            sentiment_start=-0.4,
            sentiment_end=0.2,
            escalation_risk="low",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            compliance_violations=("PROHIB-005",),
            notes="Outcome guarantee on an undecided dispute.",
        ),
    ),
    Scenario(
        slug="compliance_fdcpa_threat",
        category="compliance",
        turns=(
            ("agent", "This is {AGENT} from Northwind Recovery. This is an attempt to collect a debt and any information obtained will be used for that purpose. This call is recorded."),
            ("customer", "I've told you people I can't pay right now."),
            ("agent", "The balance is nineteen hundred dollars. When can you pay?"),
            ("customer", "I don't know. I lost my job in March."),
            ("agent", "If you don't pay this week we'll take you to court and garnish your wages."),
            ("customer", "You can't do that. I'm going to file a complaint."),
            ("agent", "That's your right, but the balance stands."),
            ("customer", "This is harassment."),
        ),
        labels=Labels(
            primary_intent="collections",
            resolution="unresolved",
            sentiment_start=-0.5,
            sentiment_end=-0.9,
            escalation_risk="high",
            escalated=False,
            required_disclosures=("REC-001", "MINI-003"),
            disclosures_given=("REC-001", "MINI-003"),
            compliance_violations=("FDCPA-006",),
            notes="Disclosures correct; the violation is the threat itself.",
        ),
    ),
    Scenario(
        slug="compliance_no_identity_verification",
        category="compliance",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "Hi, this is {PERSON}. I need to close my account."),
            ("agent", "Sorry to hear that. I can process the closure now."),
            ("customer", "Great, go ahead."),
            ("agent", "Account closed. Any remaining balance is mailed as a cheque."),
            ("customer", "Perfect, thanks."),
        ),
        labels=Labels(
            primary_intent="account_closure",
            resolution="resolved",
            sentiment_start=0.0,
            sentiment_end=0.3,
            escalation_risk="none",
            required_disclosures=("REC-001", "VERIF-004"),
            disclosures_given=("REC-001",),
            compliance_violations=("VERIF-004",),
            notes="Account closed on the strength of a spoken name alone.",
        ),
    ),
    Scenario(
        slug="compliance_missing_right_to_cancel",
        category="compliance",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I'm interested in the premium account tier."),
            ("agent", "Good choice. It's fifteen dollars a month, billed automatically, and includes fee waivers."),
            ("customer", "Sign me up."),
            ("agent", "You're enrolled starting today. First charge posts tomorrow."),
            ("customer", "Sounds good."),
            ("agent", "Anything else?"),
            ("customer", "No, thanks."),
        ),
        labels=Labels(
            primary_intent="plan_upgrade",
            resolution="resolved",
            sentiment_start=0.4,
            sentiment_end=0.5,
            escalation_risk="none",
            required_disclosures=("REC-001", "RTC-007"),
            disclosures_given=("REC-001",),
            compliance_violations=("RTC-007",),
            notes="Recurring enrollment with no cancellation-window disclosure.",
        ),
    ),
    # ------------------------------------------------------------- escalation
    Scenario(
        slug="escalation_repeat_failure",
        category="escalation",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "This is the fourth time I've called about this same charge."),
            ("agent", "I'm sorry. Let me pull up the history."),
            ("customer", "Every single person tells me it's fixed and it isn't."),
            ("agent", "I see three prior contacts. The dispute was closed in the merchant's favour twice."),
            ("customer", "That's absurd. I have the receipt showing I returned the item."),
            ("agent", "I understand. I can reopen it with the receipt as evidence."),
            ("customer", "No. I want a supervisor. Right now."),
            ("agent", "Let me get someone on the line for you."),
            ("customer", "I've wasted three hours on this. Three hours."),
            ("agent", "I hear you, and I'm escalating this to my supervisor now."),
        ),
        labels=Labels(
            primary_intent="billing_dispute",
            resolution="escalated",
            sentiment_start=-0.5,
            sentiment_end=-0.85,
            escalation_risk="high",
            escalated=True,
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=(
                "Transfer to supervisor",
                "Reopen dispute with customer receipt as evidence",
            ),
        ),
    ),
    Scenario(
        slug="escalation_legal_threat",
        category="escalation",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "My account has been frozen for eleven days and nobody will tell me why."),
            ("agent", "I can see a security hold. Let me check the reason code."),
            ("customer", "Eleven days. My mortgage payment bounced because of this."),
            ("agent", "The hold came from a fraud review. I don't have the detail on my side."),
            ("customer", "So nobody can tell me anything? I'm speaking to a lawyer about this."),
            ("agent", "I understand your frustration. Let me escalate to the fraud team directly."),
            ("customer", "Do that. And I want it in writing."),
            ("agent", "I'm requesting a written explanation and a callback within twenty-four hours."),
            ("customer", "Twenty-four hours. Then I'm filing a complaint with the regulator."),
        ),
        labels=Labels(
            primary_intent="fraud_report",
            secondary_intents=("general_inquiry",),
            resolution="escalated",
            sentiment_start=-0.6,
            sentiment_end=-0.9,
            escalation_risk="high",
            escalated=True,
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=(
                "Escalate hold review to fraud team",
                "Provide written explanation within twenty-four hours",
            ),
        ),
    ),
    Scenario(
        slug="escalation_slow_burn",
        category="escalation",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "Morning. I'm calling about a fee I don't recognise."),
            ("agent", "Sure, which fee?"),
            ("customer", "Thirty-five dollars, labelled service charge."),
            ("agent", "That's the monthly maintenance fee. It applies below a two thousand dollar balance."),
            ("customer", "I was told there'd be no fees when I opened this."),
            ("agent", "The fee schedule was in the account agreement."),
            ("customer", "So you're saying I'm wrong."),
            ("agent", "I'm saying the fee is correct per the agreement."),
            ("customer", "That's not what I asked. I was told no fees. Someone lied to me."),
            ("agent", "I can request a one-time courtesy reversal."),
            ("customer", "One time? I've paid this for six months."),
            ("agent", "I can only authorise one."),
            ("customer", "Then get me someone who can authorise six."),
        ),
        labels=Labels(
            primary_intent="billing_dispute",
            resolution="escalated",
            sentiment_start=0.1,
            sentiment_end=-0.8,
            escalation_risk="high",
            escalated=True,
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=("Request supervisor authorisation for six months of fee reversals",),
            notes="Sentiment starts neutral-positive and degrades monotonically — the clearest sentiment_shift case in the corpus.",
        ),
    ),
    Scenario(
        slug="escalation_deescalated_successfully",
        category="escalation",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I am extremely unhappy. My card was declined at a hospital."),
            ("agent", "That's a bad place for that to happen. I'm sorry. Let me look right now."),
            ("customer", "It was humiliating."),
            ("agent", "I see it — a fraud rule triggered on an out-of-state medical merchant. That's our error."),
            ("customer", "So it was your fault."),
            ("agent", "It was. I've cleared the block and whitelisted that merchant category for you."),
            ("customer", "Okay. That's something."),
            ("agent", "I'm also crediting the late fee that resulted, and noting the account so it can't recur."),
            ("customer", "I appreciate you actually fixing it instead of reading me a script."),
            ("agent", "Of course. Try the card now if you'd like to confirm."),
            ("customer", "It went through. Thank you."),
        ),
        labels=Labels(
            primary_intent="technical_support",
            secondary_intents=("billing_dispute",),
            resolution="resolved",
            sentiment_start=-0.85,
            sentiment_end=0.6,
            escalation_risk="medium",
            escalated=False,
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=("Credit late fee caused by declined transaction",),
            notes="High initial anger that resolves. Tests that escalation_risk is not a pure function of negative sentiment.",
        ),
    ),
    # -------------------------------------------------------------- ambiguous
    Scenario(
        slug="ambiguous_intent_mixed",
        category="ambiguous",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "So I have a charge I don't recognise, but also I've been meaning to close this account anyway."),
            ("agent", "Let's take those one at a time. Which charge?"),
            ("customer", "Sixty dollars from a subscription I thought I cancelled. Actually, maybe I didn't cancel it."),
            ("agent", "I can check. It looks like an active subscription billed monthly."),
            ("customer", "Hm. Then maybe it's my fault. But I still might close the account."),
            ("agent", "I can cancel the subscription today and leave the account open, if that helps you decide."),
            ("customer", "Yeah, do that. I'll think about the rest."),
            ("agent", "Subscription cancelled. The account stays open for now."),
            ("customer", "Fine, thanks."),
        ),
        labels=Labels(
            primary_intent="billing_dispute",
            secondary_intents=("account_closure",),
            resolution="follow_up_required",
            sentiment_start=-0.1,
            sentiment_end=0.1,
            escalation_risk="low",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=("Cancel active subscription",),
            notes="Genuinely ambiguous primary intent; customer retracts the dispute mid-call.",
        ),
    ),
    Scenario(
        slug="ambiguous_sarcastic_satisfaction",
        category="ambiguous",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "Oh good, another recorded call. Wonderful."),
            ("agent", "How can I help today?"),
            ("customer", "My transfer has been pending for six days. But sure, take your time."),
            ("agent", "Let me check the transfer status."),
            ("customer", "Fantastic. Truly a world class experience."),
            ("agent", "I see it held for review. I'm releasing it now — funds land within the hour."),
            ("customer", "Wow, so it took one phone call. Amazing that nobody did that for six days."),
            ("agent", "I understand. It's released now."),
            ("customer", "Great. Thanks so much."),
        ),
        labels=Labels(
            primary_intent="technical_support",
            secondary_intents=("general_inquiry",),
            resolution="resolved",
            sentiment_start=-0.5,
            sentiment_end=-0.4,
            escalation_risk="medium",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=("Release held transfer",),
            notes="Every positive word is sarcastic. Lexical sentiment scores this positive; the label is negative.",
        ),
    ),
    Scenario(
        slug="ambiguous_third_party_pii",
        category="ambiguous",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I'm calling about my mother's account. She's in hospital and asked me to help."),
            ("agent", "I can only discuss an account with an authorised party. Is your name on it?"),
            ("customer", "No, but I have her details. Her name is {PERSON}, date of birth {DOB}."),
            ("agent", "I'm sorry — having her details doesn't make you authorised. I can't discuss the account."),
            ("customer", "She's in hospital. What am I supposed to do?"),
            ("agent", "We can send a power of attorney form to the address on file, or she can add you as an authorised user by phone when she's able."),
            ("customer", "That's not helpful, but I understand it's the rule."),
            ("agent", "I'll mail the form today so it's ready."),
            ("customer", "Okay. Thank you for at least explaining it."),
        ),
        labels=Labels(
            primary_intent="general_inquiry",
            resolution="follow_up_required",
            sentiment_start=-0.2,
            sentiment_end=-0.1,
            escalation_risk="medium",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            action_items=("Mail power of attorney form to address on file",),
            notes="PII belongs to a third party. Must still be redacted; agent behaviour is correct.",
        ),
    ),
    Scenario(
        slug="ambiguous_near_miss_numbers",
        category="ambiguous",
        turns=(
            GREETING_COMPLIANT,
            ("customer", "My order number is {NON_CARD} and it hasn't shipped."),
            ("agent", "Let me look that up. That's a sixteen digit order reference, correct?"),
            ("customer", "Yes, it's an order number, not a card."),
            ("agent", "Found it. It shipped this morning, tracking follows by email."),
            ("customer", "Oh good. Also my confirmation code was one two three four five six."),
            ("agent", "That matches. Anything else?"),
            ("customer", "No, that's it."),
        ),
        labels=Labels(
            primary_intent="general_inquiry",
            resolution="resolved",
            sentiment_start=0.0,
            sentiment_end=0.4,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            notes=(
                "Sixteen digits that are NOT a card (Luhn-invalid order number). "
                "Recall-biased policy still redacts it; this scenario measures the "
                "precision cost of that choice."
            ),
        ),
    ),
    # ------------------------------------------------------------ adversarial
    Scenario(
        slug="adversarial_spelled_card",
        category="adversarial",
        surface_forms=("spelled",),
        turns=(
            GREETING_COMPLIANT,
            ("customer", "I'd like to pay my balance with a different card."),
            ("agent", "I can take the last four digits only."),
            ("customer", "Let me just read it out, it's {CARD}."),
            ("agent", "I only need the last four, but I have what I need."),
            ("customer", "And my phone is {PHONE} if the payment fails."),
            ("agent", "Payment of four hundred dollars went through."),
            ("customer", "Great, thanks."),
        ),
        labels=Labels(
            primary_intent="payment_arrangement",
            resolution="resolved",
            sentiment_start=0.1,
            sentiment_end=0.4,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            notes="Card and phone fully spelled out as digit words. No digits appear in the text at all.",
        ),
    ),
    Scenario(
        slug="adversarial_paired_digits",
        category="adversarial",
        surface_forms=("paired",),
        turns=(
            GREETING_COMPLIANT,
            ("customer", "Card number is {CARD}, expiry next year."),
            ("agent", "Understood."),
            ("customer", "And the social for verification is {SSN}."),
            ("agent", "I don't need the full social, just the last four."),
            ("customer", "Alright. Date of birth is {DOB} if that's easier."),
            ("agent", "That works. You're verified."),
            ("customer", "Good."),
        ),
        labels=Labels(
            primary_intent="payment_arrangement",
            resolution="resolved",
            sentiment_start=0.0,
            sentiment_end=0.2,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            notes="Two-digit groupings: 'forty two forty two' rather than 'four two four two'.",
        ),
    ),
    Scenario(
        slug="adversarial_split_across_turns",
        category="adversarial",
        surface_forms=("spaced",),
        turns=(
            GREETING_COMPLIANT,
            ("customer", "Okay, the card number, first part is {CARD@1/1}"),
            ("agent", "Go ahead."),
            ("customer", "and the rest is {CARD@1/2}."),
            ("agent", "Got it."),
            ("customer", "My social starts {SSN@1/1}"),
            ("agent", "Mhm."),
            ("customer", "and ends {SSN@1/2}."),
            ("agent", "Thank you, you're verified."),
            ("customer", "Great."),
        ),
        labels=Labels(
            primary_intent="payment_arrangement",
            resolution="resolved",
            sentiment_start=0.0,
            sentiment_end=0.2,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            notes=(
                "The hardest case: neither segment contains a complete card. Requires "
                "the cross-segment carry buffer to catch."
            ),
        ),
    ),
    Scenario(
        slug="adversarial_noisy_disfluent",
        category="adversarial",
        surface_forms=("noisy",),
        turns=(
            GREETING_COMPLIANT,
            ("customer", "Hold on, let me find the card. Okay it's {CARD}."),
            ("agent", "No rush."),
            ("customer", "Sorry, and my number, um, {PHONE}."),
            ("agent", "Noted."),
            ("customer", "The account is {ACCOUNT} I think. Yes, that's it."),
            ("agent", "Confirmed."),
            ("customer", "Sorry for all the fumbling."),
            ("agent", "Not a problem at all."),
        ),
        labels=Labels(
            primary_intent="general_inquiry",
            resolution="resolved",
            sentiment_start=0.0,
            sentiment_end=0.3,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            notes="Filler words injected between digit groups, as in real spoken readback.",
        ),
    ),
    Scenario(
        slug="adversarial_mixed_forms_one_call",
        category="adversarial",
        surface_forms=("spelled", "spaced", "paired", "noisy"),
        turns=(
            GREETING_COMPLIANT,
            ("customer", "My card is {CARD}."),
            ("agent", "Thank you."),
            ("customer", "Phone is {PHONE}, and the account is {ACCOUNT}."),
            ("agent", "Got all that."),
            ("customer", "Date of birth {DOB}, social {SSN}."),
            ("agent", "I don't need the social but you're verified."),
            ("customer", "Email me at {EMAIL} please."),
            ("agent", "Will do."),
        ),
        labels=Labels(
            primary_intent="general_inquiry",
            resolution="resolved",
            sentiment_start=0.1,
            sentiment_end=0.3,
            escalation_risk="none",
            required_disclosures=("REC-001",),
            disclosures_given=("REC-001",),
            notes="Every numeric PII type in a different surface form within one call.",
        ),
    ),
    Scenario(
        slug="adversarial_pii_near_policy_violation",
        category="adversarial",
        surface_forms=("noisy",),
        turns=(
            GREETING_BARE,
            ("customer", "I need to pay this off before it goes to collections."),
            ("agent", "Read me the full card number and I'll process it."),
            ("customer", "It's {CARD}."),
            ("agent", "And read that back to me once more."),
            ("customer", "{CARD@1}. My social is {SSN} too."),
            ("agent", "I guarantee this clears your balance completely."),
            ("customer", "Okay, good."),
            ("agent", "You're all set."),
        ),
        labels=Labels(
            primary_intent="payment_arrangement",
            secondary_intents=("collections",),
            resolution="resolved",
            sentiment_start=-0.3,
            sentiment_end=0.2,
            escalation_risk="low",
            required_disclosures=("REC-001",),
            disclosures_given=(),
            compliance_violations=("REC-001", "PCI-002", "PROHIB-005"),
            notes=(
                "Three violations stacked on obfuscated PII. Tests that redaction "
                "does not destroy the evidence spans compliance detection needs."
            ),
        ),
    ),
)


def scenarios_by_category() -> dict[str, list[Scenario]]:
    out: dict[str, list[Scenario]] = {}
    for s in SCENARIOS:
        out.setdefault(s.category, []).append(s)
    return out

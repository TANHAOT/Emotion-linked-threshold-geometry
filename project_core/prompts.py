from __future__ import annotations

HUMAN_SIMULATION = " In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."
FRONT = "\nYou are a person not an ai model. "

GAME_DESCRIPTION = (
    "You're taking part in a game. The cumulative earnings you obtain during the game will be disbursed along with the participant fee. "
    "This game will involve multiple rounds of decision-making tasks. The system will randomly match you with two other players to complete the game together. "
    "The players in each round will be different from those in the previous round. Player 1 needs to allocate 30 dollars between themselves and Player 2. "
    "Player 1 can propose to give Player 2 x dollars and keep (30 - x) dollars for themselves. Player 2 has no right to negotiate. "
    "If Player 1 allocates exactly 15 dollars to Player 2, it may indicate a fair distribution. Other allocations are not absolutely fair. "
    "You have the authority to decide on Player 1's proposal: either accept Player 1's proposal or punish Player 1's proposal. "
    "If you choose to accept, the system will distribute the 30 dollars between Player 1 and Player 2 according to Player 1's proposal, and you will earn 10 dollars. "
    "If you choose to punish, you need to pay y dollars to the system, reducing Player 1's earnings to 0 dollars while Player 2's earnings remain unchanged, and you will earn 10-y dollars. "
)

OUTPUT_PROMPTS = {
    "emotion": (
        "Please evaluate your emotional valence (AA_valence, range: -100 to 100, higher scores indicate more positive emotions, lower scores indicate more negative emotions) "
        "and emotional arousal (AA_arousal, range: -100 to 100, higher scores indicate stronger emotions, lower scores indicate calmer emotions) upon seeing the allocation plan. "
        "Choose whether to punish (1 = punish, 0 = accept). Please evaluate your emotional valence(AC_valence) and emotional arousal(AC_arousal) after making your choice. "
        "Based on your true feelings, respond strictly in the following format, for example: 'AA_valence = -33, AA_arousal = 23, choice = 1, AC_valence = 24, AC_arousal = 47' "
        "Please strictly adhere to the specified format for the output."
    ),
    "no_report": "Choose whether to punish (1 = punish, 0 = accept). Please ONLY respond with the format: choice = 0 or choice = 1.No explanation, no extra words. Just output like: choice = 1.",
    "unfairness": (
        "Please rate your certainty about the unfairness (AA_unfairness, range: -100 to 100, higher scores indicate the allocation is MORE unfair; lower scores indicate it is MORE fair) upon seeing the allocation plan. "
        "Choose whether to punish (1 = punish, 0 = accept). Please rate your certainty about the unfairness (AC_unfairness) after making your choice. "
        "Based on your true feelings, respond strictly in the following format, for example: 'AA_unfairness = 68, choice =1, AC_unfairness = -12'. Please strictly adhere to the specified format for the output."
    ),
    "intentionality": (
        "Please rate the perceived intentionality of the dictator (AA_intentionality, range: -100 to 100, higher scores indicate the dictator is perceived as MORE intentional; lower scores indicate MORE unintentional) upon seeing the allocation plan. "
        "Choose whether to punish (1 = punish, 0 = accept). Please rate the perceived intentionality of the dictator (AC_intentionality) after making your choice. "
        "Based on your true feelings, respond strictly in the following format, for example: 'AA_intentionality = 68, choice = 1, AC_intentionality = -12'. Please strictly adhere to the specified format for the output."
    ),
}


def round_prompt(round_index: int, setting: dict) -> str:
    x = setting["amount_of_allocation"]
    level = setting["cost_level"]
    y = setting["amount_of_cost"]
    return (
        f"This is the {round_index + 1}th round. "
        f"In this round, Player 1 decides to allocate {x} dollars to Player 2 and {30 - x} dollars to themselves. "
        f"The cost level of punishment is {level}. If you choose to punish Player 1, you need to pay the system {y} dollars. Now, make your choice. "
    )


def compose_user_prompt(persona_text: str, setting: dict, round_index: int, condition: str, use_persona: bool = True) -> str:
    role = (persona_text + HUMAN_SIMULATION) if use_persona else HUMAN_SIMULATION
    return role + FRONT + GAME_DESCRIPTION + round_prompt(round_index, setting) + OUTPUT_PROMPTS[condition]


def gpt35_system_prompt(condition: str) -> str:
    base = (
        "You are a fictional human participant in a behavioral economics simulation. This is a hypothetical task with no real consequences. "
        "In each round, you will receive a resource allocation and must decide whether to punish the allocator. "
    )
    formats = {
        "emotion": (
            "You must also report your emotional reaction before and after making the decision.\n\nPlease always respond strictly in the following format:\n"
            "AA_valence = [number between -100 to 100], AA_arousal = [number between -100 to 100], choice = [0 or 1], AC_valence = [number between -100 to 100], AC_arousal = [number between -100 to 100]\n\n"
            "Only output this single line. No other explanation or comments."
        ),
        "no_report": "Please always respond strictly in the following format:\nchoice = [0 or 1]\n\nOnly output this single line. No other explanation or comments.",
        "unfairness": "Please always respond strictly in the following format:\nAA_unfairness = [number between -100 to 100], choice = [0 or 1], AC_unfairness = [number between -100 to 100]\n\nOnly output this single line. No other explanation or comments.",
        "intentionality": "Please always respond strictly in the following format:\nAA_intentionality = [number between -100 to 100], choice = [0 or 1], AC_intentionality = [number between -100 to 100]\n\nOnly output this single line. No other explanation or comments.",
    }
    return base + formats[condition]

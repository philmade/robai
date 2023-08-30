from robai.base import AIRobot
from robai.in_out import ChatMessage
from typing import List, Callable, Self
from robai.languagemodels import OpenAIChatCompletion
from robai.examples.memory import SummaryRobotMemory
from robai.utility import pprint_color


# UTILITY
def split_text_into_token_chunks(text: str, chunk_length_limit: int) -> List[str]:
    """
    Split the text into chunks based on an estimated token count.
    """
    average_tokens_per_word = 1.5
    words = text.split()
    words_per_chunk = int(chunk_length_limit / average_tokens_per_word)

    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i : i + words_per_chunk]))

    return chunks


# PRE-CALL CHAIN
def split_text_into_chunks(
    memory: SummaryRobotMemory,
) -> SummaryRobotMemory:
    to_summarise = memory.input_model
    chunk_length_limit = memory.chunk_length_limit
    if not memory.chunks:
        chunks = split_text_into_token_chunks(to_summarise, chunk_length_limit)
        memory.chunks = chunks
        memory.current_chunk_index = 0
        memory.total_chunks = len(chunks)
    return memory


def provide_context_for_chunk(
    memory: SummaryRobotMemory,
) -> SummaryRobotMemory:
    to_summarize = memory.chunks.pop(0)  # Pop the first chunk as the current content
    current_chunk_num = memory.current_chunk_index + 1
    context = f"""
    Task: {memory.purpose} You performing your task. You are at section {current_chunk_num} of {memory.total_chunks}. \n\n
    Here's what you've summarised so far {memory.summaries}\n\n
    You must now summarise this chunk of text and we'll add it to the summary so far: {to_summarize}\n\n
    """
    memory.instructions_for_ai = [ChatMessage(role="user", content=context)]
    return memory


# POST-CALL CHAIN
def append_summary_and_check_complete(
    memory: SummaryRobotMemory,
) -> SummaryRobotMemory:
    if not hasattr(memory, "summaries"):
        memory.summaries = []
    # APPEND THE SUMMARY WE HAVE RECEIVED TO THE LIST OF SUMMARIES
    memory.summaries.append(memory.ai_response.content)
    memory.current_chunk_index += 1
    # CHECK IF WE ARE DONE
    if memory.current_chunk_index == memory.total_chunks:
        # There are no more chunks to summarise,
        # we are done. memory.set_complete() means the robot will not go back to pre-call.
        memory.set_complete()
    else:
        # There are more chunks to summarise. We are not done.
        # The robot sends everything back to pre-call and
        # we'll summarise the next chunk.
        # Note that in pre-call we pop() the chunks, so when they're processed they are gone.
        pass

    return memory


# ASSEMBLE THE CHAINS
summary_pre_call_chain = [split_text_into_chunks, provide_context_for_chunk]

summary_post_call_chain = [append_summary_and_check_complete]


class SummaryRobot(AIRobot):
    def __init__(
        self,
        ai_model: OpenAIChatCompletion = OpenAIChatCompletion(),
        memory: SummaryRobotMemory = SummaryRobotMemory(
            purpose="""Please judge this statement as one of two words: True/False. Then justify why"""
        ),
        pre_call_chain: List[Callable] = summary_pre_call_chain,
        post_call_chain: List[Callable] = summary_post_call_chain,
        **kwargs,
    ) -> Self:
        self.memory = memory
        self.ai_model = ai_model
        self.pre_call_chain = pre_call_chain
        self.post_call_chain = post_call_chain


if __name__ == "__main__":
    robot = SummaryRobot()
    text_to_summarise = """Opening Statement
            Mr. Chairman, Ranking Members, and Congressmen,
            Thank you, I am happy to be here. This is an important issue, and I am grateful for your time.
            My name is David Charles Grusch. I was an intelligence officer for 14 years, both in the US Air
            Force (USAF) at the rank of Major and most recently, from 2021-2023, at the National GeospatialIntelligence Agency at the GS-15 civilian level, which is the military equivalent of a full-bird
            Colonel. I was my agency’s co-lead in Unidentified Anomalous Phenomena (UAP) and transmedium object analysis, as well as reporting to UAP Task Force (UAPTF) and eventually the AllDomain Anomaly Resolution Office (AARO).
            I became a Whistleblower, through a PPD-19 Urgent Concern filing with the Intelligence
            Community Inspector General (ICIG), following concerning reports from multiple esteemed and
            credentialed current and former military and Intelligence Community individuals that the US
            Government is operating with secrecy - above Congressional oversight - with regards to UAPs.
            My testimony is based on information I have been given by individuals with a longstanding track
            record of legitimacy and service to this country – many of whom also shared compelling
            evidence in the form of photography, official documentation, and classified oral testimony.
            I have taken every step I can to corroborate this evidence over a period of 4 years and to do my
            due diligence on the individuals sharing it, and it is because of these steps that I believe strongly
            in the importance of bringing this information before you.
            I am driven by a commitment to truth and transparency, rooted in our inherent duty to uphold the
            United States Constitution and protect the American People. I am asking Congress to hold our
            Government to this standard and thoroughly investigate these claims. But as I stand here under
            oath now, I am speaking to the facts as I have been told them.
            In the USAF, in my National Reconnaissance Office (NRO) reservist capacity, I was a member
            of the UAPTF from 2019-2021. I served in the NRO Operations Center on the director’s briefing 
            staff, which included the coordination of the Presidential Daily Brief (PDB) and supporting
            contingency operations.
            In 2019, the UAPTF director tasked me to identify all Special Access Programs & Controlled
            Access Programs (SAPs/CAPs) we needed to satisfy our congressionally mandated mission.
            At the time, due to my extensive executive-level intelligence support duties, I was cleared to
            literally all relevant compartments and in a position of extreme trust in both my military and
            civilian capacities.
            I was informed, in the course of my official duties, of a multi-decade UAP crash retrieval and
            reverse engineering program to which I was denied access to those additional read-on’s.
            I made the decision based on the data I collected, to report this information to my superiors and
            multiple Inspectors General, and in effect become a whistleblower.
            As you know, I have suffered retaliation for my decision. But I am hopeful that my actions will
            ultimately lead to a positive outcome of increased transparency.
            Thank you. I am happy to answer your questions.
            Closing Statement
            It is with a heavy heart and a determined spirit that I stand, under oath, before you today, having
            made the decision based on the data I collected, and reported, to provide this information to the
            committee. I am driven in this duty by a conviction to expose what I viewed as a grave
            congressional oversight issue and a potential abuse of executive branch authorities.
            This endeavor was not born out of malice or dissatisfaction, but from an unwavering commitment
            to truth and transparency, an endeavor rooted in our inherent duty to uphold the United States
            Constitution, protect the American People, and seek insights into this matter that have the potential
            to redefine our understanding of the world. 
            In an era, fraught with division and discord, our exploration into the UAP subject seems to resonate
            with an urgency and fascination that transcends political, social, and geographical boundaries. A
            democratic process must be adhered to when evaluating the data and it is our collective
            responsibility to ensure that public involvement is encouraged and respected. Indeed, the future of
            our civilization and our comprehension of humanity’s place on earth and in the cosmos depends
            on the success of this very process.
            It is my hope that the revelations we unearth through investigations of the Non-Human Reverse
            Engineering Programs I have reported will act as an ontological (earth-shattering) shock, a catalyst
            for a global reassessment of our priorities. As we move forward on this path, we might be poised
            to enable extraordinary technological progress in a future where our civilization surpasses the
            current state-of-the-art in propulsion, material science, energy production and storage.
            The knowledge we stand to gain should spur us toward a more enlightened and sustainable future,
            one where collective curiosity is ignited, and global cooperation becomes the norm, rather than the
            exception.
            Thank You ."""
    pprint_color("ROBOT IS SUMMARISING TEXT")
    pprint_color(text_to_summarise)
    input_model: str = text_to_summarise
    pprint_color("PROCESSING...")
    memory = robot.process(input_model)
    pprint_color("DONE")
    pprint_color("SUMMARY:")
    pprint_color("=========")
    pprint_color(memory.ai_response.content)

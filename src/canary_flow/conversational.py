#!/usr/bin/env python
from crewai import Flow
from crewai.experimental.conversational import ConversationConfig, ConversationState
from crewai.flow import listen

from canary_flow.crews.research_crew.research_crew import ResearchCrew

GREETING_TRIGGERS = {
    "good afternoon",
    "good evening",
    "good morning",
    "hello",
    "hey",
    "hi",
    "hiya",
    "howdy",
}

SEARCH_TRIGGERS = {
    "find",
    "latest",
    "look up",
    "news",
    "recent",
    "research",
    "search",
    "source",
    "sources",
    "up-to-date",
    "web",
}


class ResearchState(ConversationState):
    topic: str = ""
    report: str = ""


@ConversationConfig(llm="gpt-5.4-mini")
class ResearchFlow(Flow[ResearchState]):
    conversational = True

    def route_turn(self, context: dict) -> str:
        message = (self.state.current_user_message or "").strip().lower()
        if message in {"bye", "exit", "goodbye", "quit"}:
            return "end"
        if message.startswith("save") and self.state.report:
            return "save_report"
        if any(trigger in message for trigger in SEARCH_TRIGGERS):
            return "internet_search"
        if message in GREETING_TRIGGERS or any(
            message.startswith(f"{trigger} ") or message.startswith(f"{trigger},")
            for trigger in GREETING_TRIGGERS
        ):
            return "greeting"
        return "converse"

    @listen("greeting")
    def send_greeting(self) -> str:
        """Generic Greeting messages should return fast."""
        reply = "How can I help you today? You have internet search access"
        self.append_assistant_message(reply)
        return reply

    @listen("internet_search")
    def conduct_web_research(self) -> str:
        """Fresh web research, current news, and source-backed lookups."""
        topic = (self.state.current_user_message or "").strip()
        self.state.topic = topic

        print(f"Conducting web research on: {topic}")
        result = ResearchCrew().crew().kickoff(inputs={"topic": topic})

        reply = getattr(result, "raw", None) or str(result)
        self.state.report = reply
        self.append_agent_result(
            "internet_search",
            reply,
            visibility="private",
            metadata={"route": "internet_search", "topic": topic},
        )
        self.append_assistant_message(reply)
        return reply

    @listen("save_report")
    def save_research_report(self) -> str:
        """Save the most recent research report to disk."""
        filename = f"research_report_{self.state.topic.replace(' ', '_')[:30]}.txt"
        with open(filename, "w") as report_file:
            report_file.write(f"Research Report: {self.state.topic}\n")
            report_file.write("=" * 60 + "\n\n")
            report_file.write(self.state.report)

        reply = f"Saved the latest research report to {filename}."
        self.append_assistant_message(reply)
        return reply


def kickoff() -> None:
    ResearchFlow().chat(defer_trace_finalization=True)


if __name__ == "__main__":
    print(ResearchFlow.flow_definition().to_yaml())
    # kickoff()

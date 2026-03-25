#!/usr/bin/env python
from pydantic import BaseModel

from crewai.flow import Flow, human_feedback, listen, start

from canary_flow.create_vertex_config import create_llm
from canary_flow.crews.research_crew.research_crew import ResearchCrew

_HITL_LLM = create_llm()


class ResearchState(BaseModel):
    topic: str = ""
    report: str = ""


class ResearchFlow(Flow[ResearchState]):
    @start()
    def select_research_topic(self):
        """Select a random research topic or use a predefined one"""
        print("Selecting research topic")
        self.state.topic = "lorenze jay hernandez"
        print(f"Selected topic: {self.state.topic}")

    @listen(select_research_topic)
    def conduct_web_research(self):
        """Use ResearchCrew with EXASearchTool to research the topic"""
        print(f"Conducting web research on: {self.state.topic}")
        result = ResearchCrew().crew().kickoff(inputs={"topic": self.state.topic})

        print("Research completed")
        self.state.report = result.raw
        self.remember(result.raw)
        return result.token_usage

    @listen(conduct_web_research)
    @human_feedback(
        message=(
            "Review the draft research report shown above. "
            "Your free-form input will be mapped to: approve (save as-is) or revise (append your notes to the report)."
        ),
        emit=["approve", "revise"],
        llm=_HITL_LLM,
        default_outcome="approve",
    )
    def review_report(self):
        """Pause for human review before saving."""
        return self.state.report

    def _write_report_file(self) -> None:
        print("Saving research report")
        filename = f"research_report_{self.state.topic.replace(' ', '_')[:30]}.txt"
        with open(filename, "w") as f:
            f.write(f"Research Report: {self.state.topic}\n")
            f.write("=" * 60 + "\n\n")
            f.write(self.state.report)
        print(f"Report saved to: {filename}")

    @listen("approve")
    def save_research_report(self):
        """Save the research report to a file (approved as-is)."""
        self._write_report_file()

    @listen("revise")
    def save_research_report_with_notes(self):
        """Append human revision notes, then save."""
        hf = self.last_human_feedback
        if hf and hf.feedback.strip():
            self.state.report += (
                "\n\n---\nHuman revision notes:\n" + hf.feedback.strip()
            )
        self._write_report_file()


def kickoff():
    research_flow = ResearchFlow()
    research_flow.kickoff()


if __name__ == "__main__":
    # Run the research flow by default
    # Change to kickoff() to run the poem flow
    kickoff()

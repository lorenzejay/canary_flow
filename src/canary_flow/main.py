#!/usr/bin/env python
from random import randint, choice

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from canary_flow.crews.research_crew.research_crew import ResearchCrew


class ResearchState(BaseModel):
    topic: str = ""
    report: str = ""


class ResearchFlow(Flow[ResearchState]):
    @start()
    def select_research_topic(self):
        """Select a random research topic or use a predefined one"""
        print("Selecting research topic")
        topics = [
            "Latest developments in artificial intelligence",
            "Current trends in renewable energy",
            "Recent advances in quantum computing",
            "Impact of remote work on productivity",
            "Emerging cybersecurity threats in 2024",
        ]
        self.state.topic = choice(topics)
        print(f"Selected topic: {self.state.topic}")

    @listen(select_research_topic)
    def conduct_web_research(self):
        """Use ResearchCrew with EXASearchTool to research the topic"""
        print(f"Conducting web research on: {self.state.topic}")
        result = ResearchCrew().crew().kickoff(inputs={"topic": self.state.topic})

        print("Research completed")
        self.state.report = result.raw
        return result.token_usage

    @listen(conduct_web_research)
    def save_research_report(self):
        """Save the research report to a file"""
        print("Saving research report")
        filename = f"research_report_{self.state.topic.replace(' ', '_')[:30]}.txt"
        with open(filename, "w") as f:
            f.write(f"Research Report: {self.state.topic}\n")
            f.write("=" * 60 + "\n\n")
            f.write(self.state.report)
        print(f"Report saved to: {filename}")


def kickoff():
    research_flow = ResearchFlow()
    research_flow.kickoff()


if __name__ == "__main__":
    # Run the research flow by default
    # Change to kickoff() to run the poem flow
    kickoff()

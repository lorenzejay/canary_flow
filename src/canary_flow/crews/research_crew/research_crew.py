from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import EXASearchTool
from typing import List


@CrewBase
class ResearchCrew:
    """Research Crew for conducting web research using EXA"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["research_analyst"],  # type: ignore[index]
            tools=[EXASearchTool()],
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"],  # type: ignore[index]
        )

    @task
    def conduct_research(self) -> Task:
        return Task(
            config=self.tasks_config["conduct_research"],  # type: ignore[index]
        )

    @task
    def write_report(self) -> Task:
        return Task(
            config=self.tasks_config["write_report"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

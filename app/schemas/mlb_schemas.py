from pydantic import BaseModel, Field


class MLBTeams(BaseModel):
    hometeam: str = Field(default="Boston Red Sox", description="Home team name")
    awayteam: str = Field(default="Chicago Cubs", description="Away team name")


class MLBSingleTeam(BaseModel):
    name: str = Field(default="Boston Red Sox", description="Team name")
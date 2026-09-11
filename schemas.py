from pydantic import BaseModel, Field
from typing import List, Optional


class LeadershipMember(BaseModel):
    """Schema for a single leadership/team member."""
    name: str = Field(description="Full name of the person")
    role: str = Field(description="Job title or role, e.g., CEO, CTO, Founder")
    linkedin_url: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL if found on the page. Return None if not found."
    )


class CompanyIntelligence(BaseModel):
    """Main schema for the extracted company data."""
    
    company_overview: str = Field(
        description="A concise 2-sentence summary of what the company does."
    )
    
    target_audience: str = Field(
        description="Who the product is built for, e.g., 'Developers building backend applications'."
    )
    
    contact_emails: List[str] = Field(
        default_factory=list,
        description="List of generic or public emails found on the site (e.g., contact@, sales@, support@)."
    )
    
    leadership: List[LeadershipMember] = Field(
        default_factory=list,
        description="List of key leadership or team members found on the site."
    )
    
    confidence_score: float = Field(
        description="A score between 0.0 and 1.0 indicating how complete and reliable the extracted data is.",
        ge=0.0,
        le=1.0
    )

    hallucination_score: float = Field(
        description="A score between 0.0 and 1.0 indicating how much of the extracted data was inferred or hallucinated rather than directly quoted from the text.",
        ge=0.0,
        le=1.0
    )

    source_url: Optional[str] = Field(
        default=None,
        description="The URL where this data was scraped from."
    )
"""Pydantic schemas: these define the EXACT JSON shape the LLM must return."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------- Invoice ----------------
class LineItem(BaseModel):
    description: str
    quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: str = Field(description="Format YYYY-MM-DD")
    due_date: Optional[str] = Field(default=None, description="Format YYYY-MM-DD or null")
    vendor_name: str
    customer_name: str
    currency: str = Field(description="ISO code, e.g. INR, USD")
    items: List[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    discount: float = Field(default=0, ge=0)
    tax: float = Field(default=0, ge=0)
    total: float = Field(ge=0)

    @field_validator("invoice_date", "due_date")
    @classmethod
    def check_date(cls, v):
        if v is None:
            return v
        datetime.strptime(v, "%Y-%m-%d")  # raises ValueError if the format is wrong
        return v

    @model_validator(mode="after")
    def check_total(self):
        expected = self.subtotal - self.discount + self.tax
        if abs(expected - self.total) > 0.5:
            raise ValueError(
                f"total ({self.total}) must equal subtotal - discount + tax ({expected:.2f})"
            )
        return self


# ---------------- Resume ----------------
class Education(BaseModel):
    degree: str
    institution: str
    year: Optional[str] = None


class Experience(BaseModel):
    role: str
    company: str
    duration: Optional[str] = None
    highlights: List[str] = []


class Resume(BaseModel):
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = []
    education: List[Education] = []
    experience: List[Experience] = []

    @field_validator("email")
    @classmethod
    def check_email(cls, v):
        if v is not None and "@" not in v:
            raise ValueError("email must contain '@'")
        return v


SCHEMAS = {"invoice": Invoice, "resume": Resume}
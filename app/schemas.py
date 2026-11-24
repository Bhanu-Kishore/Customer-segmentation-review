"""Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    age: int = Field(..., ge=18, le=100, description="Customer age in years")
    annual_income: float = Field(
        ..., ge=1_000, le=1_000_000, description="Annual income in USD"
    )
    spending_score: float = Field(
        ..., ge=0, le=100, description="Spending score from 0 to 100"
    )
    purchase_frequency: int = Field(
        ..., ge=0, le=365, description="Purchases per year"
    )


class PredictRequest(CustomerFeatures):
    pass


class CustomerCreate(CustomerFeatures):
    name: str | None = Field(default=None, max_length=80)


class CustomerUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = Field(default=None, max_length=80)
    age: int | None = Field(default=None, ge=18, le=100)
    annual_income: float | None = Field(default=None, ge=1_000, le=1_000_000)
    spending_score: float | None = Field(default=None, ge=0, le=100)
    purchase_frequency: int | None = Field(default=None, ge=0, le=365)


class CustomerResponse(BaseModel):
    id: int
    name: str
    age: int
    annual_income: float
    spending_score: float
    purchase_frequency: int
    segment: int
    segment_name: str
    created_at: str
    updated_at: str


class CustomerListResponse(BaseModel):
    items: list[CustomerResponse]
    total: int
    page: int
    page_size: int
    pages: int

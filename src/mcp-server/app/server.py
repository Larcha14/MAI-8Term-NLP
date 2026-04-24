from fastmcp import FastMCP

mcp = FastMCP("Credit MCP Server")


@mcp.tool
def calculate_credit_score(
    age: int,
    income: float,
    has_overdues: bool,
    has_higher_education: bool,
    married: bool,
    employment_years: float,
) -> dict:
    """
    Calculate a simple demonstration credit score for a client.

    Args:
        age: Client age in years.
        income: Monthly income.
        has_overdues: Whether the client had overdue payments before.
        has_higher_education: Whether the client has higher education.
        married: Whether the client is married.
        employment_years: Employment duration in years.

    Returns:
        Dictionary with score and explanation.
    """
    score = 50

    if 25 <= age <= 55:
        score += 10
    if income >= 100000:
        score += 20
    elif income >= 60000:
        score += 10
    if not has_overdues:
        score += 15
    else:
        score -= 25
    if has_higher_education:
        score += 5
    if married:
        score += 5
    if employment_years >= 3:
        score += 10
    elif employment_years < 1:
        score -= 10

    score = max(0, min(100, score))
    return {"score": score}


@mcp.tool
def assess_credit_risk(
    income: float,
    has_overdues: bool,
    employment_years: float,
    married: bool,
) -> dict:
    """
    Assess a simple demonstration credit risk level.

    Args:
        income: Monthly income.
        has_overdues: Whether the client had overdue payments before.
        employment_years: Employment duration in years.
        married: Whether the client is married.

    Returns:
        Dictionary with risk label.
    """
    if has_overdues and income < 60000:
        return {"risk": "high"}
    if employment_years < 1 and income < 80000:
        return {"risk": "high"}
    if income >= 100000 and not has_overdues and employment_years >= 2:
        return {"risk": "low"}
    if married and income >= 80000 and not has_overdues:
        return {"risk": "low"}
    return {"risk": "medium"}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=9000)
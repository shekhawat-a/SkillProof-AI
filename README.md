# SkillProof-AI

## Backend interview candidate IDs

Technical interviews use candidate IDs from the JDS skills workbook; behavioral
interviews use IDs from the separate SDS personality workbook. The datasets do
not share candidate IDs, so they must not be treated as the same people.

Start the backend from `backend` with `uvicorn app.main:app --reload`. To list
available IDs for a given interview type, call
`GET /api/interview/candidates?interview_type=behavioral` or use
`interview_type=technical`. The response identifies the dataset type and
returns its candidate IDs. Install dependencies from `backend/requirements.txt`;
the Groq package is required by both interview services.
"""
Project 3: Build a PR Review workflow with multiple agents
"""

import dotenv
import os
import asyncio

from typing import Any
from github import Github, Auth
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context

dotenv.load_dotenv()

# Create the LLM object with the model name, key and proxy address from .env
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

#Set up connection to github

git = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN"))) if os.getenv("GITHUB_TOKEN") else None

# Repository ("username/repo-name") and PR number, set by GitHub Actions in ci.yml
full_repo_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

if git is not None:
    repo = git.get_repo(full_repo_name)

#Context tools: gather pr details, file contents, and commit details
def get_pr_details(pr_number: int) -> dict:
    """Fetch details about a pull request given its number: user, title, body, diff URL, state, head commit SHA, and all commit SHAs."""
    pull_request = repo.get_pull(pr_number)

    commit_SHAs = []
    commits = pull_request.get_commits()
    for c in commits:
        commit_SHAs.append(c.sha)

    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_SHAs": commit_SHAs,
    }

def get_file_contents(file_path: str) -> str:
    """Fetch the contents of a file from the repository given its path, for example 'app/models.py'."""
    file_content = repo.get_contents(file_path)
    return file_content.decoded_content.decode("utf-8")

def get_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Fetch the files changed in a commit given its SHA: filename, status, additions, deletions, changes, and patch (diff)."""
    commit = repo.get_commit(commit_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files

# State tools: save the gathered context and the draft comment into the shared workflow state
async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Useful for adding the gathered contexts to the state."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "State updated with gathered contexts."

async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for adding the draft comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment."

# Post the final review to the PR on GitHub as a comment-only review
async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for adding the final review to the state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review."


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post the final review comment to the GitHub pull request with the given number."""
    pull_request = repo.get_pull(pr_number)
    review = pull_request.create_review(body=comment, event="COMMENT")
    return f"Review posted: {review.html_url}"

#Convert functions to a tool schema that the agent can work with
pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
commit_details_tool = FunctionTool.from_defaults(get_commit_details)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)

#Build the ReAct Agent
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository: PR details, changed files, and any requested repository files.",
    tools=[pr_details_tool, file_contents_tool, commit_details_tool, add_context_to_state],
    system_prompt="""You are the context gathering agent. When gathering context, you MUST gather:
  - The details: author, title, body, diff_url, state, and head_sha;
  - Changed files;
  - Any requested for files;
Once you gather the requested info, save it with the add_context_to_state tool.
Then you MUST hand control back to the CommentorAgent.
""",
    can_handoff_to=["CommentorAgent"],
)

#Build the Commentor Agent
commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull request review comment.",
    tools=[add_comment_to_state],
    system_prompt="""You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the ContextAgent.
 - Once you have written the review, save it with the add_comment_to_state tool.
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
""",
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

# Agent that checks the draft review, asks for rewrites if needed, and posts the final review to GitHub
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft PR comment from the CommentorAgent, requests rewrites if it falls short, and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt="""You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
The review must:
   - Be a ~200-300 word review in markdown format.
   - Specify what is good about the PR.
   - Say whether the author followed ALL contribution rules, and what is missing.
   - Have notes on test availability for new functionality. If there are new models, note whether there are migrations for them.
   - Have notes on whether new endpoints were documented.
   - Have suggestions on which lines could be improved upon, with those lines quoted.
If the review does not meet these criteria, you must ask the CommentorAgent to rewrite and address these concerns.
When you are satisfied, save the review with the add_final_review_to_state tool, then post it to GitHub.
""",
    can_handoff_to=["CommentorAgent"],
)

#Multi-agent workflow
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)

# Read a question, run the agent on it, and print each agent name, model output, and tool result as they happen
async def main():
    # Build the query from the PR number, since there's no one to type input on GitHub
    query = f"Write a review for PR number {pr_number} and post the final review to GitHub."

    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")
if __name__ == "__main__":
    asyncio.run(main())
    git.close()
import os
import requests
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI  # Updated for OpenRouter
import git  # For git operations

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeComment:
    """Represents a comment on a specific piece of code"""
    file_path: str
    line_number: int
    comment: str
    suggestion: Optional[str] = None
    category: str = "general"  # e.g., bug, style, performance, security

@dataclass
class PRReview:
    """Container for all review feedback on a PR"""
    general_comments: List[str]
    code_comments: List[CodeComment]
    score: Optional[float] = None  # Overall quality score (0-100)

class GitClient(ABC):
    """Abstract base class for git server clients"""
    
    @abstractmethod
    def get_pr_diff(self, repo_url: str, pr_id: str) -> str:
        """Get the diff for a specific PR"""
        pass
    
    @abstractmethod
    def post_comment(self, repo_url: str, pr_id: str, comment: CodeComment) -> bool:
        """Post a comment to the PR"""
        pass
    
    @abstractmethod
    def get_pr_details(self, repo_url: str, pr_id: str) -> Dict[str, Any]:
        """Get PR metadata (title, description, etc.)"""
        pass

class GitHubClient(GitClient):
    """GitHub implementation of GitClient"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def get_pr_diff(self, repo_url: str, pr_id: str) -> str:
        # Extract owner and repo from URL
        parts = repo_url.rstrip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        # Get diff from the specific endpoint
        diff_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_id}.diff"
        diff_response = requests.get(diff_url, headers=self.headers)
        diff_response.raise_for_status()
        
        return diff_response.text
    
    def post_comment(self, repo_url: str, pr_id: str, comment: CodeComment) -> bool:
        parts = repo_url.rstrip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_id}/comments"
        
        # GitHub needs position in the diff, not absolute line number
        # This is a simplified approach - in production you'd need to calculate position
        payload = {
            "body": f"**{comment.category.upper()}**: {comment.comment}\n\nSuggestion: {comment.suggestion}",
            "path": comment.file_path,
            "line": comment.line_number
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        return response.status_code == 201
    
    def get_pr_details(self, repo_url: str, pr_id: str) -> Dict[str, Any]:
        parts = repo_url.rstrip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.json()

class GitLabClient(GitClient):
    """GitLab implementation of GitClient"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('GITLAB_TOKEN')
        self.headers = {'Private-Token': self.token}
    
    def get_pr_diff(self, repo_url: str, pr_id: str) -> str:
        # GitLab calls PRs "merge requests"
        project_id = self._get_project_id(repo_url)
        url = f"https://gitlab.com/api/v4/projects/{project_id}/merge_requests/{pr_id}/changes"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        changes = response.json()
        return changes.get('changes', [])
    
    def post_comment(self, repo_url: str, pr_id: str, comment: CodeComment) -> bool:
        project_id = self._get_project_id(repo_url)
        url = f"https://gitlab.com/api/v4/projects/{project_id}/merge_requests/{pr_id}/notes"
        
        payload = {
            "body": f"{comment.category.upper()}: {comment.comment}\nSuggestion: {comment.suggestion}",
            "position": {
                "base_sha": "TODO",  # Need to get from MR details
                "start_sha": "TODO",  # Need to get from MR details
                "head_sha": "TODO",  # Need to get from MR details
                "position_type": "text",
                "new_path": comment.file_path,
                "new_line": comment.line_number
            }
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        return response.status_code == 201
    
    def _get_project_id(self, repo_url: str) -> str:
        # In a real implementation, we'd need to encode the project path
        # and look up the project ID
        encoded_path = repo_url.replace('https://gitlab.com/', '').replace('/', '%2F')
        url = f"https://gitlab.com/api/v4/projects/{encoded_path}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.json().get('id')
    
    def get_pr_details(self, repo_url: str, pr_id: str) -> Dict[str, Any]:
        project_id = self._get_project_id(repo_url)
        url = f"https://gitlab.com/api/v4/projects/{project_id}/merge_requests/{pr_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.json()

class BitbucketClient(GitClient):
    """Bitbucket implementation of GitClient"""
    
    def __init__(self, username: Optional[str] = None, token: Optional[str] = None):
        self.username = username or os.getenv('BITBUCKET_USERNAME')
        self.token = token or os.getenv('BITBUCKET_TOKEN')
        self.auth = (self.username, self.token)
    
    def get_pr_diff(self, repo_url: str, pr_id: str) -> str:
        workspace, repo_slug = self._extract_workspace_and_repo(repo_url)
        url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}/diff"
        
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        
        return response.text
    
    def post_comment(self, repo_url: str, pr_id: str, comment: CodeComment) -> bool:
        workspace, repo_slug = self._extract_workspace_and_repo(repo_url)
        url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}/comments"
        
        # Bitbucket needs the specific diff position
        payload = {
            "content": {"raw": f"{comment.category.upper()}: {comment.comment}\nSuggestion: {comment.suggestion}"},
            "anchor": {
                "path": comment.file_path,
                "line": comment.line_number,
                "line_type": "added"
            }
        }
        
        response = requests.post(url, auth=self.auth, json=payload)
        return response.status_code == 201
    
    def _extract_workspace_and_repo(self, repo_url: str) -> tuple:
        parts = repo_url.rstrip('/').split('/')
        return parts[-2], parts[-1]
    
    def get_pr_details(self, repo_url: str, pr_id: str) -> Dict[str, Any]:
        workspace, repo_slug = self._extract_workspace_and_repo(repo_url)
        url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}"
        
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        
        return response.json()

class CodeAnalyzer:
    """Analyzes code changes and provides feedback"""
    
    def __init__(self, openrouter_api_key: Optional[str] = None, 
                 site_url: Optional[str] = None, 
                 site_name: Optional[str] = None):
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.site_url = site_url or os.getenv('OPENROUTER_SITE_URL')
        self.site_name = site_name or os.getenv('OPENROUTER_SITE_NAME')
        
        if self.openrouter_api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )
    
    def analyze_diff(self, diff: str, pr_details: Dict[str, Any]) -> PRReview:
        """Analyze the diff and return review comments"""
        general_comments = []
        code_comments = []
        
        # Use AI analysis if API key is available
        if self.openrouter_api_key:
            ai_review = self._analyze_with_ai(diff, pr_details)
            general_comments.extend(ai_review.general_comments)
            code_comments.extend(ai_review.code_comments)
        
        # Add rule-based analysis
        rule_based_review = self._rule_based_analysis(diff)
        general_comments.extend(rule_based_review.general_comments)
        code_comments.extend(rule_based_review.code_comments)
        
        # Calculate score
        score = self._calculate_score(general_comments, code_comments)
        
        return PRReview(general_comments, code_comments, score)
    
    def _analyze_with_ai(self, diff: str, pr_details: Dict[str, Any]) -> PRReview:
        """Use OpenRouter API to analyze code changes"""
        try:
            prompt = self._create_analysis_prompt(diff, pr_details)
            
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                extra_body={},
                model="x-ai/grok-code-fast-1",
                messages=[
                    {"role": "system", "content": "You are a code review assistant. Provide constructive feedback on code changes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return self._parse_ai_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return PRReview([], [])
    
    def _create_analysis_prompt(self, diff: str, pr_details: Dict[str, Any]) -> str:
        """Create a prompt for AI analysis"""
        title = pr_details.get('title', '')
        description = pr_details.get('description', '') or pr_details.get('body', '')
        
        return f"""
        Please review the following code changes from a pull request:
        
        PR Title: {title}
        PR Description: {description}
        
        Code Changes:
        {diff}
        
        Provide feedback in the following JSON format:
        {{
            "general_comments": ["list of general comments"],
            "code_comments": [
                {{
                    "file_path": "path/to/file",
                    "line_number": 123,
                    "comment": "specific feedback",
                    "suggestion": "suggested improvement",
                    "category": "bug|style|performance|security|general"
                }}
            ]
        }}
        
        Focus on:
        1. Code quality and maintainability
        2. Potential bugs or issues
        3. Security vulnerabilities
        4. Performance optimizations
        5. Consistency with coding standards
        6. Readability and documentation
        
        Be constructive and specific in your feedback.
        """
    
    def _parse_ai_response(self, response: str) -> PRReview:
        """Parse the AI response into a PRReview object"""
        try:
            # Extract JSON from response (in case there's additional text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            
            data = json.loads(json_str)
            
            general_comments = data.get('general_comments', [])
            code_comments_data = data.get('code_comments', [])
            
            code_comments = []
            for comment_data in code_comments_data:
                code_comments.append(CodeComment(
                    file_path=comment_data.get('file_path', ''),
                    line_number=comment_data.get('line_number', 0),
                    comment=comment_data.get('comment', ''),
                    suggestion=comment_data.get('suggestion', ''),
                    category=comment_data.get('category', 'general')
                ))
            
            return PRReview(general_comments, code_comments)
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return PRReview([], [])
    
    def _rule_based_analysis(self, diff: str) -> PRReview:
        """Perform rule-based analysis on the diff"""
        general_comments = []
        code_comments = []
        
        # Simple rule-based checks
        lines = diff.split('\n')
        current_file = None
        
        for i, line in enumerate(lines):
            # Track current file
            if line.startswith('+++ b/'):
                current_file = line[6:]
                continue
            
            # Check for large files (heuristic)
            if line.startswith('+') and len(line) > 200:
                code_comments.append(CodeComment(
                    file_path=current_file or 'unknown',
                    line_number=i,
                    comment="Line is very long, consider breaking it up for readability",
                    category="style"
                ))
            
            # Check for TODO comments
            if line.startswith('+') and 'TODO' in line.upper():
                code_comments.append(CodeComment(
                    file_path=current_file or 'unknown',
                    line_number=i,
                    comment="TODO comment found in new code",
                    suggestion="Consider creating an issue instead of leaving TODO comments",
                    category="general"
                ))
            
            # Check for commented code
            if line.startswith('+') and line.strip().startswith('//') and len(line.strip()) > 5:
                code_comments.append(CodeComment(
                    file_path=current_file or 'unknown',
                    line_number=i,
                    comment="Commented code found",
                    suggestion="Remove commented code or explain why it's needed",
                    category="general"
                ))
        
        return PRReview(general_comments, code_comments)
    
    def _calculate_score(self, general_comments: List[str], code_comments: List[CodeComment]) -> float:
        """Calculate a quality score based on the review findings"""
        # Base score
        score = 100.0
        
        # Deduct points based on issues found
        severity_weights = {
            "bug": 10.0,
            "security": 15.0,
            "performance": 7.0,
            "style": 2.0,
            "general": 3.0
        }
        
        for comment in code_comments:
            score -= severity_weights.get(comment.category, 5.0)
        
        # Deduct for general comments
        score -= len(general_comments) * 5.0
        
        # Ensure score is within bounds
        return max(0.0, min(100.0, score))

class PRReviewAgent:
    """Main agent that coordinates the PR review process"""
    
    def __init__(self, git_client: GitClient, code_analyzer: CodeAnalyzer):
        self.git_client = git_client
        self.code_analyzer = code_analyzer
    
    def review_pr(self, repo_url: str, pr_id: str, post_comments: bool = False) -> PRReview:
        """Review a PR and optionally post comments"""
        logger.info(f"Reviewing PR {pr_id} from {repo_url}")
        
        # Get PR details and diff
        try:
            pr_details = self.git_client.get_pr_details(repo_url, pr_id)
            diff = self.git_client.get_pr_diff(repo_url, pr_id)
        except Exception as e:
            logger.error(f"Failed to get PR details: {e}")
            raise
        
        # Analyze the code changes
        review = self.code_analyzer.analyze_diff(diff, pr_details)
        
        # Post comments if requested
        if post_comments:
            for comment in review.code_comments:
                try:
                    success = self.git_client.post_comment(repo_url, pr_id, comment)
                    if not success:
                        logger.warning(f"Failed to post comment for {comment.file_path}:{comment.line_number}")
                except Exception as e:
                    logger.error(f"Error posting comment: {e}")
        
        return review

def create_git_client(repo_url: str, **kwargs) -> GitClient:
    """Factory function to create the appropriate git client based on repo URL"""
    if 'github.com' in repo_url:
        return GitHubClient(**kwargs)
    elif 'gitlab.com' in repo_url:
        return GitLabClient(**kwargs)
    elif 'bitbucket.org' in repo_url:
        return BitbucketClient(**kwargs)
    else:
        raise ValueError(f"Unsupported git server: {repo_url}")

# Example usage
if __name__ == "__main__":
    # Configuration
    REPO_URL = "https://github.com/ageron/handson-ml3.git"  # Replace with your repo URL
    PR_ID = "1"  # Replace with your PR ID
    POST_COMMENTS = False  # Set to True to actually post comments
    
    # Create clients
    git_client = create_git_client(REPO_URL)
    code_analyzer = CodeAnalyzer(
        openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
        site_url=os.getenv('OPENROUTER_SITE_URL'),
        site_name=os.getenv('OPENROUTER_SITE_NAME')
    )
    
    # Create and run agent
    agent = PRReviewAgent(git_client, code_analyzer)
    review = agent.review_pr(REPO_URL, PR_ID, POST_COMMENTS)
    
    # Print results
    print(f"PR Review Score: {review.score}/100")
    print("\nGeneral Comments:")
    for comment in review.general_comments:
        print(f"- {comment}")
    
    print("\nCode Comments:")
    for comment in review.code_comments:
        print(f"{comment.file_path}:{comment.line_number} [{comment.category}]")
        print(f"  Comment: {comment.comment}")
        if comment.suggestion:
            print(f"  Suggestion: {comment.suggestion}")
        print()
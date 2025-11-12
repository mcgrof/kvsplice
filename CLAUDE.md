# Claude AI Assistant Preferences

## Git Commit Practices

### Commit Structure
- Make small, atomic commits - one logical change per commit
- Each commit should be functional and not break the build
- Run code formatter (black for Python) after each change
- Run scripts/fix_whitespace_issues.py always on all files
- Test that code runs successfully before committing

### Commit Messages
- **MANDATORY**: Always use this exact format for ALL commits:
  ```
  file.py: brief description of change

  Detailed explanation of what was changed and why.
  Include technical details about the implementation.

  Generated-by: Claude AI
  Signed-off-by: Luis Chamberlain <mcgrof@kernel.org>
  ```

- **LINE LENGTH**: Maximum 70 characters per line in commit messages
  - Subject line (first line): 70 characters max
  - Body paragraphs: 70 characters max per line
  - Ensures proper display in git log, email patches, and terminal output
- **CRITICAL**: Never use "🤖 Generated with [Claude Code]" or "Co-Authored-By: Claude"
- **REQUIRED**: Every commit MUST have both "Generated-by: Claude AI" and "Signed-off-by: Luis Chamberlain <mcgrof@kernel.org>"
- **NO EXCEPTIONS**: This format is mandatory for ALL commits, no matter how small
- **STYLE**: Be terse and to the point. NO shopping-list style bullet points. Write in paragraphs explaining the change, rationale, and technical details concisely. Avoid verbose enumeration unless absolutely necessary for clarity.

### Development Workflow
1. Make a single focused change
2. Run `black` formatter on Python files
3. Test that the code runs without errors
4. **If architectural changes**: Run `make check` to validate
5. Commit with detailed message
6. Repeat for next change

Architectural changes include:
- New attention or MLP mechanisms
- Modified forward/backward pass logic
- Changes to model patching or wrapper classes
- New ablation steps or configurations

## Code Style

### Python
- Use `black` formatter for all Python code
- Follow PEP 8 conventions (handled by black)
- No manual formatting - always use black

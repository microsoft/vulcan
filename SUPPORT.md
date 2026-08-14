# Support

## How to file issues and get help

This project uses [GitHub Issues](https://github.com/microsoft/vulcan/issues) to track bugs
and feature requests. Please search the existing issues before filing a new one, to avoid
duplicates.

**For bugs**, a report is much easier to act on when it includes:

- the config you ran (with any API key removed) and the exact command
- which provider and model (`llm.provider` and `llm.model`)
- the stage that failed, and the tail of `<output_base>/logs/<category>/<step>.log`
- what you expected to happen instead

A stage can report success while every model call failed, so if the output looks empty
rather than wrong, check the `llm_calls=` count in that stage log first — it is the
honest signal.

**For questions about using VULCAN**, open a
[GitHub Discussion](https://github.com/microsoft/vulcan/discussions) or a question-labelled
issue. Before asking, [`docs/`](docs/) covers most of the sharp edges — in particular
[`docs/providers.md`](docs/providers.md) for provider setup and capability differences,
and [`docs/configuration.md`](docs/configuration.md) for what each config key does.

## Security issues

**Do not report security vulnerabilities through public GitHub issues.** See
[SECURITY.md](SECURITY.md) for the reporting process.

Note that VULCAN executes model-generated code without a sandbox by design — see the
warning at the top of the [README](README.md) before
reporting that as a vulnerability.

## Microsoft support policy

Support for this project is limited to the resources listed above. It is a research
project, provided as-is under the [MIT License](LICENSE), and is not covered by Microsoft
Customer Service & Support (CSS).

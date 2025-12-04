
Setup uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
```
Clone and cd
```bash
git clone https://github.com/israel-adewuyi/inno_swe_reasoner.git && cd inno_swe_reasoner
```

```bash
uv sync && uv run pre-commit install
```
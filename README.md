# tabs
Keeping up with the current understanding and practices of AI and application security. 

## Running

Install: `pip install -e ".[dev]"`

Run ingestion manually: `tabs ingest`

To run daily via cron, add a line like this to your crontab (`crontab -e`),
adjusting the path and using an absolute path to the `tabs` executable from
your virtualenv (find it with `which tabs` after activating the env):

    0 6 * * * cd /Users/spencer/projects/tabs && /path/to/venv/bin/tabs ingest >> data/ingest.log 2>&1

#!/bin/bash
set -e

echo "=== OpenCode Installer ==="

# Step 1: Install Node.js (if not present)
if ! command -v node &> /dev/null; then
    echo "[1/5] Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
else
    echo "[1/5] Node.js already installed: $(node --version)"
fi

# Step 2: Install OpenCode CLI
echo "[2/5] Installing OpenCode CLI..."
npm install -g opencode-ai

# Step 3: Create config directory
echo "[3/5] Configuring settings..."
CONFIG_DIR="$HOME/.config/opencode"
mkdir -p "$CONFIG_DIR"

# Step 4: Write opencode.json
SETTINGS_FILE="$CONFIG_DIR/opencode.json"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Warning: GITHUB_TOKEN environment variable is not set."
    echo "  GitHub MCP server will be disabled."
    echo "  Usage: GITHUB_TOKEN=ghp_xxx bash install-opencode.sh"
    GITHUB_MCP_ENABLED="false"
    GITHUB_TOKEN_VALUE=""
else
    GITHUB_MCP_ENABLED="true"
    GITHUB_TOKEN_VALUE="$GITHUB_TOKEN"
    echo "  -> Using GITHUB_TOKEN from environment variable"
fi

cat > "$SETTINGS_FILE" << EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "mcp": {
    "context7": {
      "command": [
        "npx",
        "-y",
        "@upstash/context7-mcp"
      ],
      "enabled": true,
      "type": "local"
    },
    "github": {
      "enabled": ${GITHUB_MCP_ENABLED},
      "headers": {
        "Authorization": "Bearer ${GITHUB_TOKEN_VALUE}"
      },
      "type": "remote",
      "url": "https://api.githubcopilot.com/mcp/"
    },
    "playwright": {
      "command": [
        "npx",
        "@playwright/mcp@latest"
      ],
      "enabled": true,
      "type": "local"
    }
  },
  "plugin": [
    "superpowers@git+https://github.com/obra/superpowers.git",
    "@tarquinen/opencode-dcp@latest"
  ]
}
EOF

# Step 5: Write dcp.jsonc and install code graph
npm i -g @colbymchenry/codegraph
codegraph install --yes
codegraph init -i

DCP_FILE="$CONFIG_DIR/dcp.jsonc"
echo "[4/5] Writing dcp.jsonc config..."
cat > "$DCP_FILE" << 'DCPEOF'
{
    "$schema": "https://raw.githubusercontent.com/Opencode-DCP/opencode-dynamic-context-pruning/master/dcp.schema.json",
    "enabled": true,
    "autoUpdate": true,
    "debug": false,
    "pruneNotification": "detailed",
    "pruneNotificationType": "chat",
    "commands": {
        "enabled": true,
        "protectedTools": [],
    },
    "manualMode": {
        "enabled": false,
        "automaticStrategies": true,
    },
    "turnProtection": {
        "enabled": false,
        "turns": 4,
    },
    "experimental": {
        "allowSubAgents": false,
        "customPrompts": false,
    },
    "protectedFilePatterns": [],
    "compress": {
        "mode": "range",
        "permission": "allow",
        "showCompression": false,
        "summaryBuffer": true,
        "maxContextLimit": "80%",
        "minContextLimit": "50%",
        "nudgeFrequency": 5,
        "iterationNudgeThreshold": 15,
        "nudgeForce": "soft",
        "protectedTools": [],
        "protectTags": false,
        "protectUserMessages": false,
    },
    "strategies": {
        "deduplication": {
            "enabled": true,
            "protectedTools": [],
        },
        "purgeErrors": {
            "enabled": true,
            "turns": 4,
            "protectedTools": [],
        },
    },
}
DCPEOF

# Step 6: Verify installation
echo "[5/6] Verifying installation..."
if command -v opencode &> /dev/null; then
    echo "[6/6] OpenCode installed: $(opencode --version 2>/dev/null || echo 'CLI ready')"
else
    echo "Warning: opencode command not found in PATH"
fi

echo ""
echo "=== Installation Complete ==="
echo "Settings written to: $SETTINGS_FILE"
echo "DCP config written to: $DCP_FILE"
echo ""

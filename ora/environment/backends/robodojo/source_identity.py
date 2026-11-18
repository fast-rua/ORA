"""Source-pin admission: strict production, advisory diagnostic replay only."""
import hashlib
import subprocess

from ora.contracts.toolkit_protocol import identity_warning


class SourceIdentityAudit:
    """Report measured identities without changing sources or physics settings.

    Only host-created diagnostic configs set replay_identity_warnings. The
    production caller retains the same error class and source-pin requirements.
    """

    def __init__(self, config, error_type):
        self.config, self.error_type = config, error_type
        self.diagnostic = config.get("replay_identity_warnings") is True
        self.warnings = []

    def mismatch(self, message, field, recorded, current):
        if not self.diagnostic:
            raise self.error_type(message)
        self.warnings.append({**identity_warning(field, recorded, current), "message": message})

    def sources(self, roots, required_keys, revision):
        actual, required = {}, {}
        for name, path in roots.items():
            expected = str(self.config.get(required_keys[name], ""))
            if len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
                self.mismatch(f"missing exact source pin: {required_keys[name]}", required_keys[name], expected, None)
            required[name] = expected
            try:
                actual[name] = revision(path)
            except self.error_type as error:
                self.mismatch(str(error), f"source_revisions.{name}", expected, None)
                actual[name] = None
                continue
            if actual[name] != expected:
                self.mismatch(f"source revision mismatch: {name}", f"source_revisions.{name}", expected, actual[name])
            # Ignore untracked caches/assets; tracked modifications alter semantics.
            dirty = subprocess.run(["git", "-C", str(path), "diff", "HEAD", "--", "."],
                                   capture_output=True, check=True)
            if dirty.stdout:
                self.mismatch(f"tracked source is modified: {name}", f"source_dirty.{name}", False,
                              {"patch_sha256": hashlib.sha256(dirty.stdout).hexdigest()})
        return actual, required

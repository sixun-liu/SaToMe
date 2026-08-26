# Upstream Revisions

SaToMe integrates with two independently maintained backbones. Public release
artifacts must preserve their copyright and attribution notices.

| Backbone | Repository | Pinned revision | License |
|---|---|---|---|
| LaViDa | https://github.com/jacklishufan/LaViDa | `24220c0077f403d24ca52968b6a7c55b687d5c2f` | Apache-2.0 |
| LLaDA-o | https://github.com/ML-GSAI/LLaDA-o | `00b40b4123c730eaf765bba439f63dd1d453d152` | Apache-2.0 |

The LLaDA-o README at the pinned revision still shows the historical
`GSAI-ML/LLaDA-o` clone URL. The accessible public repository is currently
hosted under `ML-GSAI/LLaDA-o`.

Both repositories are pinned as git submodules under `third_party/`. SaToMe
keeps upstream history untouched: small integration diffs live under
`integrations/<backbone>/patches/` and are applied explicitly by bootstrap
scripts. Modified files retain upstream notices and add a SaToMe modification
notice.

# TODO — Figure Terminology Updates (Comment 3 Package)

The following **image assets were not modified** in this release. They still contain legacy **点—线—面 / Point–Line–Surface** wording. Update manually before final paper/open-source publication.

## Figures requiring manual replacement

| File | Issue | Suggested replacement text |
|------|-------|---------------------------|
| `image/fig2.png` | Title/caption uses 点-线-面 framework | account--community--network hierarchy |
| `image/fig3.png` | Caption uses 面级隐藏成员发现 | network-level hidden-member expansion |

## README references (if full README is published separately)

| Location | Legacy term | Preferred term |
|----------|-------------|----------------|
| `README.md` line ~5 | 点—线—面（Point–Line–Surface） | account--community--network hierarchy |
| `README.md` architecture section | 点级 / 线级 / 面级 | account level / community level / network level |
| `README.md` alt text on fig2 | MLAGDet 点-线-面框架 | MLAGDet account-community-network framework |

## Script / output filenames (optional cleanup)

| Path | Note |
|------|------|
| `experiment/phase2/alpha_sensitivity/intermediate/point_level_scores.csv` | Uses `point_level` in filename; consider `account_level_scores.csv` in future refactor |

## Action

- [ ] Regenerate or edit `fig2.png` and `fig3.png` with updated English/Chinese labels
- [ ] Update main repository `README.md` architecture section (outside this lightweight zip)
- [ ] Confirm manuscript figure captions match code terminology

# Documentation Index - Wetting Front Tracker

Complete documentation for the Wetting Front Tracker project. Start here to find what you need!

## 📚 Documentation Suite

### For All Users

1. **[PROJECT_README.md](PROJECT_README.md)** 📖
   - **Start here!** Complete project overview
   - Installation instructions
   - Usage examples
   - Scientific background
   - 📄 ~8,000 words

2. **[QUICK_START.md](QUICK_START.md)** 🚀
   - **Get running in 5 minutes**
   - Step-by-step setup
   - First run instructions
   - Common issues and fixes
   - 📄 ~4,000 words

### For Data Users

3. **[DATA_FORMATS.md](DATA_FORMATS.md)** 📊
   - All input/output formats
   - File schemas and specifications
   - Data validation rules
   - Example files
   - 📄 ~5,000 words

4. **[COLORING_EXAMPLES.md](COLORING_EXAMPLES.md)** 🎨
   - How polygon colors work
   - Priority rules explained
   - Example scenarios
   - Color interpretation guide
   - 📄 ~2,500 words

### For Developers

5. **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** 💻
   - Architecture overview
   - Code organization
   - Adding new features
   - Testing guidelines
   - Performance optimization
   - 📄 ~5,500 words

6. **[CODE_REFERENCE.md](CODE_REFERENCE.md)** 📝
   - Key code snippets
   - Function signatures
   - Quick lookup reference
   - 📄 ~1,500 words

### Recent Changes

7. **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** 🆕
   - Latest feature: Average LWC coloring
   - Detailed change documentation
   - Migration guide
   - 📄 ~2,000 words

## 🎯 Finding What You Need

### "I'm brand new to this project"
→ Start with **[QUICK_START.md](QUICK_START.md)** to get running  
→ Then read **[PROJECT_README.md](PROJECT_README.md)** for full context

### "I need to understand the data"
→ Read **[DATA_FORMATS.md](DATA_FORMATS.md)** for complete specifications  
→ Check **[COLORING_EXAMPLES.md](COLORING_EXAMPLES.md)** to understand results

### "I want to modify the code"
→ Read **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** for architecture  
→ Use **[CODE_REFERENCE.md](CODE_REFERENCE.md)** for quick lookups

### "I want to know what changed"
→ See **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** for recent updates

### "I have a specific question"
→ Use the search function (Ctrl+F / Cmd+F) in any document  
→ Check the Table of Contents in each file

## 📁 Files Included

### Modified Python Files (with LWC coloring feature)
- **main.py** - Orchestrator with LWC integration
- **wet_front_tracker.py** - Added avg_lwc_above_weak()
- **plotting.py** - Updated coloring logic

### Documentation Files
- **README.md** - (This file) Master index
- **PROJECT_README.md** - Complete project documentation
- **QUICK_START.md** - Getting started guide
- **DATA_FORMATS.md** - Data specifications
- **DEVELOPER_GUIDE.md** - Development documentation
- **CODE_REFERENCE.md** - Code snippets
- **COLORING_EXAMPLES.md** - Color logic examples
- **CHANGES_SUMMARY.md** - Recent changes

### Configuration
- **requirements.txt** - Python dependencies

## 🚀 Quick Links

**Get Started:**
- [Installation Steps](QUICK_START.md#installation-steps)
- [First Run](QUICK_START.md#first-run)
- [Common Issues](QUICK_START.md#common-first-time-issues)

**Understand Output:**
- [Color Meanings](COLORING_EXAMPLES.md#priority-rules)
- [Map Features](PROJECT_README.md#summary-map-summary_maphtml)
- [Analysis Metrics](PROJECT_README.md#analysis-metrics)

**For Developers:**
- [Architecture](DEVELOPER_GUIDE.md#architecture-overview)
- [Add New Features](DEVELOPER_GUIDE.md#adding-new-features)
- [Code Examples](CODE_REFERENCE.md)

## 📊 Quick Reference

### Color Codes
| Color | Meaning | Priority |
|-------|---------|----------|
| 🔴 Red | LWC > 3% above LOC | **HIGHEST** |
| 🟡 Yellow | LWC 1-3% OR 48-72h to LOC | High |
| 🟧 Orange | 24-48h to LOC | Medium |
| 🟥 Dark Red | 0-24h to LOC (imminent) | High |
| 🔵 Blue | LOC reached (past) | Low |
| ⚪ Gray | No data | Lowest |

### Command Line Usage
```bash
# Default run
python -m src.wetting_front_tracker.main

# Specific date
python -m src.wetting_front_tracker.main --date 2025-05-15

# Custom date range
python -m src.wetting_front_tracker.main \
    --date 2025-05-15 \
    --start-date 2025-05-01 \
    --end-date 2025-05-31

# More workers
python -m src.wetting_front_tracker.main --workers 16
```

## 🔄 What's New

**Version 1.1 (November 2025):**
- ✅ Added average LWC above LOC coloring
- ✅ Priority-based color system (water content > time)
- ✅ Updated map legend
- ✅ Enhanced risk visualization

See [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) for complete details.

## 📖 Recommended Reading Order

### For End Users
1. [QUICK_START.md](QUICK_START.md) - Setup and first run
2. [PROJECT_README.md](PROJECT_README.md) - Full overview
3. [COLORING_EXAMPLES.md](COLORING_EXAMPLES.md) - Interpret results
4. [DATA_FORMATS.md](DATA_FORMATS.md) - Understand data

### For Developers
1. [PROJECT_README.md](PROJECT_README.md) - Project context
2. [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Architecture
3. [CODE_REFERENCE.md](CODE_REFERENCE.md) - Code patterns
4. [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - Recent work

### For Analysts
1. [QUICK_START.md](QUICK_START.md) - Get running
2. [COLORING_EXAMPLES.md](COLORING_EXAMPLES.md) - Understand colors
3. [DATA_FORMATS.md](DATA_FORMATS.md) - Data specs
4. [PROJECT_README.md § Analysis Metrics](PROJECT_README.md#analysis-metrics)

## 🆘 Need Help?

**Common Solutions:**
- Installation issues → [QUICK_START.md § Common Issues](QUICK_START.md#common-first-time-issues)
- Understanding output → [COLORING_EXAMPLES.md](COLORING_EXAMPLES.md)
- Data format questions → [DATA_FORMATS.md](DATA_FORMATS.md)
- Code problems → [DEVELOPER_GUIDE.md § Debugging](DEVELOPER_GUIDE.md#debugging-tips)

**Still Stuck?**
1. Check logs: `cat wetting_front_tracker.log`
2. Enable debug mode in code
3. Open an issue with error details

## 📊 Documentation Stats

- **Total Documents:** 8
- **Total Words:** ~28,500
- **Total Lines:** ~2,580
- **Topics Covered:** 63

## ⚖️ License

[Specify your license here]

## 📮 Contact

- **Issues:** [Open a GitHub issue]
- **Questions:** [Contact development team]
- **Contributions:** See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

---

**Version:** 1.1 (November 2025)  
**Authors:** Ron Simenhois, Itai  
**Status:** Production Ready

**Happy Analyzing! 🏔️❄️**

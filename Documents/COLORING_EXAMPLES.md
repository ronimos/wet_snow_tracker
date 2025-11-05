# Coloring Examples

This document shows how the new polygon coloring logic works with various combinations of `time_to_loc` and `avg_lwc_above_loc` values.

## Priority Rules
1. **LWC > 3%** → RED (highest priority)
2. **LWC 1-3%** → YELLOW (high priority)
3. **Time-based** → Various colors (fallback)

## Example Scenarios

### Scenario 1: High Water Content Overrides Imminent Warning
```
time_to_loc = 12 hours (normally darkred - imminent)
avg_lwc_above_loc = 4.5%
RESULT: RED (water content overrides time)
REASON: Water content >3% is highest priority
```

### Scenario 2: Moderate Water Content 
```
time_to_loc = 36 hours (normally orange - 24-48h)
avg_lwc_above_loc = 2.1%
RESULT: YELLOW (water content overrides time)
REASON: Water content 1-3% is high priority
```

### Scenario 3: Low Water Content - Use Time Color
```
time_to_loc = 60 hours
avg_lwc_above_loc = 0.8%
RESULT: YELLOW (time-based, 48-72h)
REASON: Water content <1%, so use time_to_loc coloring
```

### Scenario 4: No Water Data - Use Time Color
```
time_to_loc = 6 hours
avg_lwc_above_loc = None
RESULT: DARKRED (time-based, 0-24h imminent)
REASON: No water content data, use time_to_loc coloring
```

### Scenario 5: High Water Content, Recent Event
```
time_to_loc = -12 hours (past, normally red)
avg_lwc_above_loc = 5.2%
RESULT: RED (water content overrides time)
REASON: Water content >3% has highest priority
NOTE: Both would be red anyway, but water content is checked first
```

### Scenario 6: Moderate Water Content, Far Future
```
time_to_loc = 65 hours (normally yellow - 48-72h)
avg_lwc_above_loc = 1.5%
RESULT: YELLOW (water content overrides time)
REASON: Water content 1-3%
NOTE: Both would be yellow anyway, but water content is checked first
```

### Scenario 7: Very High Water Content
```
time_to_loc = 100 hours (normally gray - unknown)
avg_lwc_above_loc = 8.3%
RESULT: RED (water content overrides time)
REASON: Extremely high water content >3%
```

### Scenario 8: Edge Case - Exactly 1%
```
time_to_loc = 30 hours (normally orange - 24-48h)
avg_lwc_above_loc = 1.0%
RESULT: YELLOW (water content overrides time)
REASON: 1.0% is included in the 1-3% range (>= 1.0)
```

### Scenario 9: Edge Case - Exactly 3%
```
time_to_loc = 30 hours (normally orange - 24-48h)
avg_lwc_above_loc = 3.0%
RESULT: YELLOW (water content overrides time)
REASON: 3.0% is NOT >3%, so it falls in 1-3% range
```

### Scenario 10: Edge Case - Just Over 3%
```
time_to_loc = 30 hours (normally orange - 24-48h)
avg_lwc_above_loc = 3.01%
RESULT: RED (water content overrides time)
REASON: 3.01% > 3.0%, triggers red condition
```

## Color Combinations Table

| time_to_loc | avg_lwc | Without LWC Logic | With LWC Logic | Override? |
|-------------|---------|-------------------|----------------|-----------|
| 12h         | 0.5%    | darkred           | darkred        | No        |
| 12h         | 2.0%    | darkred           | **yellow**     | **Yes**   |
| 12h         | 4.0%    | darkred           | **red**        | **Yes**   |
| 36h         | 0.5%    | orange            | orange         | No        |
| 36h         | 2.0%    | orange            | **yellow**     | **Yes**   |
| 36h         | 4.0%    | orange            | **red**        | **Yes**   |
| 60h         | 0.5%    | yellow            | yellow         | No        |
| 60h         | 2.0%    | yellow            | yellow         | No*       |
| 60h         | 4.0%    | yellow            | **red**        | **Yes**   |
| -12h        | 0.5%    | red               | red            | No        |
| -12h         | 2.0%    | red               | **yellow**     | **Yes**   |
| -12h         | 4.0%    | red               | red            | No*       |

*No visual change, but logic path is different

## Integration Points

### Where Colors Are Assigned
```python
# In create_folium_map() function:
final_gdf['color'] = final_gdf.apply(
    lambda row: get_polygon_color(
        row['time_to_loc'], 
        row.get('avg_lwc_above_loc')
    ),
    axis=1
)
```

### Where Colors Are Used
1. **Map polygons** - fill color based on assigned value
2. **Legend** - shows meaning of each color
3. **GeoJSON export** - color stored in properties for external use

## Testing Tips

1. **Check threshold boundaries:**
   - Test with 0.99%, 1.00%, 1.01%
   - Test with 2.99%, 3.00%, 3.01%

2. **Check missing data:**
   - Test with `avg_lwc_above_loc = None`
   - Test with `avg_lwc_above_loc = NaN`

3. **Check time ranges:**
   - Test all time_to_loc categories
   - Verify water content always overrides when present

4. **Visual verification:**
   - High water areas should stand out as red
   - Moderate water should be yellow
   - Areas with time-based colors should have low/no water content

## Common Questions

**Q: What if both conditions give the same color?**
A: The water content logic is checked first, but the result is the same. Example: time_to_loc=-10h (red) and avg_lwc=4% (red) both result in red.

**Q: Can a polygon be orange with high water content?**
A: No. If avg_lwc > 3%, it will be red. If 1-3%, it will be yellow. Orange only appears when avg_lwc < 1% or is missing.

**Q: What happens if weak layer detection fails?**
A: If no weak layer is found, avg_lwc_above_loc will be None, and the polygon uses time_to_loc coloring only.

**Q: Are the thresholds (1% and 3%) configurable?**
A: Currently they are hardcoded in `get_polygon_color()`. To change them, modify the function in plotting.py.

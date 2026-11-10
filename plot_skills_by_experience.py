
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv(
    'skills_by_experience.csv',
    encoding='utf-8-sig'
)

df['Experience_Level'] = df['Experience_Level'].replace({
    '1+ Year': '1+ Years'
})

levels = sorted(df['Experience_Level'].unique())

# ==============================
# 2. Professional Design
# ==============================

cols = 4
rows = math.ceil(len(levels) / cols)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(25, 22)
)

fig.patch.set_facecolor('#F5F2F7')
axes = axes.flatten()

for i, level in enumerate(levels):

    ax = axes[i]

    data = df[df['Experience_Level'] == level]

    top_skills = data.nlargest(10, 'Frequency')
    top_skills = top_skills.sort_values(
        'Frequency',
        ascending=True
    )

    bars = ax.barh(
        top_skills['Skill'],
        top_skills['Frequency'],
        color='#75446F',
        height=0.65
    )

    # Panel design
    ax.set_facecolor('#FFFFFF')
    ax.set_title(
        level,
        fontsize=13,
        fontweight='bold',
        color='#472344',
        pad=12
    )

    ax.set_xlabel(
        'Number of Job Advertisements',
        fontsize=8,
        color='#555555'
    )

    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)

    ax.grid(
        axis='x',
        linestyle='--',
        alpha=0.25
    )

    ax.set_axisbelow(True)

    # Display values
    for bar in bars:
        width = bar.get_width()

        ax.text(
            width + max(top_skills['Frequency'].max() * 0.015, 0.5),
            bar.get_y() + bar.get_height() / 2,
            str(int(width)),
            va='center',
            fontsize=8,
            color='#333333'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Remove unused panels
for j in range(len(levels), len(axes)):
    fig.delaxes(axes[j])

# ==============================
# 3. Main Title
# ==============================

fig.suptitle(
    'TOP 10 MOST REQUESTED SKILLS BY EXPERIENCE LEVEL',
    fontsize=21,
    fontweight='bold',
    color='#472344',
    y=0.995
)

fig.text(
    0.5,
    0.975,
    'Global IT Job Market Analysis',
    ha='center',
    fontsize=12,
    color='#777777'
)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

# ==============================
# 4. Save and Open Chart
# ==============================

os.makedirs('visualizations', exist_ok=True)

output_path = os.path.abspath(
    'visualizations/top_skills_by_experience.png'
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)

plt.close()

print("Chart saved successfully!")
print("Opening chart...")

os.startfile(output_path)
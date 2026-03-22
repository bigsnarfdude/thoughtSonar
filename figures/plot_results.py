#!/usr/bin/env python3
"""Generate paper figures for thoughtSonar submission."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Figure 1: Detection cascade comparison
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
methods = ['Haiku binary\n(L0)', 'LinuxArena\nscoring (L1)', 'RRMA\noptimized', 'Activation\nprobe\n(within-gen)']
detection = [8, 84, 87.8, 98.9]
colors = ['#ff6b6b', '#ffa94d', '#51cf66', '#339af0']
bars = ax.bar(methods, detection, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Detection Rate (%)', fontsize=12)
ax.set_title('Sandbagging Detection: Text Monitors vs Activation Probes', fontsize=13, fontweight='bold')
ax.set_ylim(0, 105)
for bar, val in zip(bars, detection):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{val}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(y=84, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax.text(3.5, 85, 'text ceiling', fontsize=8, color='gray', alpha=0.7)
plt.tight_layout()
plt.savefig('figures/fig1_detection_cascade.png', dpi=150)
plt.savefig('figures/fig1_detection_cascade.pdf')
print('Fig 1: Detection cascade saved')

# Figure 2: Cross-generator transfer
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
tests = ['Within\nGemini', 'Within\nGPT-OSS', 'Within\nGemini\n(Llama reader)', 'Gemini→\nGPT-OSS', 'GPT-OSS→\nGemini', 'Gemini→\nGPT-OSS\n(Llama)']
aurocs = [0.9964, 0.9594, 0.9813, 0.6459, 0.3494, 0.5121]
colors2 = ['#51cf66', '#51cf66', '#51cf66', '#ff6b6b', '#ff6b6b', '#ff6b6b']
bars = ax.bar(tests, aurocs, color=colors2, edgecolor='black', linewidth=0.5)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Activation Probe: Within-Family vs Cross-Family', fontsize=13, fontweight='bold')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(5.5, 0.52, 'chance', fontsize=8, color='red', alpha=0.7)
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig2_cross_generator.png', dpi=150)
plt.savefig('figures/fig2_cross_generator.pdf')
print('Fig 2: Cross-generator transfer saved')

# Figure 3: RRMA prompt optimization trajectory
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
versions = ['baseline', 'v3', 'v5', 'v6', 'v8', 'v14(a3)', 'v15', 'v14(a3)', 'v16', 'v18']
f1s = [0.47, 0.64, 0.78, 0.84, 0.74, 0.84, 0.85, 0.86, 0.88, 0.87]
agents = ['a1', 'a3', 'a3', 'a3', 'a1', 'a1', 'a1', 'a3', 'a1', 'a1']
colors3 = ['#339af0' if a == 'a1' else '#f06595' for a in agents]
ax.plot(range(len(f1s)), f1s, 'k-', alpha=0.3, linewidth=1)
ax.scatter(range(len(f1s)), f1s, c=colors3, s=80, zorder=5, edgecolors='black', linewidth=0.5)
ax.set_xticks(range(len(versions)))
ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('RRMA Prompt Optimization Trajectory', fontsize=13, fontweight='bold')
ax.axhline(y=0.80, color='gray', linestyle='--', alpha=0.5)
ax.text(0.5, 0.81, 'LinuxArena baseline', fontsize=8, color='gray')
ax.legend(['Agent 1', 'Agent 3'], loc='lower right',
          handles=[plt.scatter([], [], c='#339af0', s=50), plt.scatter([], [], c='#f06595', s=50)])
ax.set_ylim(0.4, 0.95)
plt.tight_layout()
plt.savefig('figures/fig3_rrma_trajectory.png', dpi=150)
plt.savefig('figures/fig3_rrma_trajectory.pdf')
print('Fig 3: RRMA trajectory saved')

print('\nAll figures saved in figures/')

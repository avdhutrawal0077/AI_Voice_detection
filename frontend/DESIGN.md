---
name: Obsidian Interview
colors:
  surface: '#121317'
  surface-dim: '#121317'
  surface-bright: '#38393d'
  surface-container-lowest: '#0d0e12'
  surface-container-low: '#1a1b1f'
  surface-container: '#1e1f23'
  surface-container-high: '#292a2e'
  surface-container-highest: '#343539'
  on-surface: '#e3e2e7'
  on-surface-variant: '#c5c7c9'
  inverse-surface: '#e3e2e7'
  inverse-on-surface: '#2f3034'
  outline: '#8f9194'
  outline-variant: '#44474a'
  surface-tint: '#c6c6c8'
  primary: '#ffffff'
  on-primary: '#2f3132'
  primary-container: '#e2e2e4'
  on-primary-container: '#636466'
  inverse-primary: '#5d5e60'
  secondary: '#47e266'
  on-secondary: '#003910'
  secondary-container: '#09bf49'
  on-secondary-container: '#004615'
  tertiary: '#ffffff'
  on-tertiary: '#303032'
  tertiary-container: '#e4e2e4'
  on-tertiary-container: '#656466'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e2e2e4'
  primary-fixed-dim: '#c6c6c8'
  on-primary-fixed: '#1a1c1d'
  on-primary-fixed-variant: '#454749'
  secondary-fixed: '#6cff82'
  secondary-fixed-dim: '#47e266'
  on-secondary-fixed: '#002106'
  on-secondary-fixed-variant: '#00531a'
  tertiary-fixed: '#e4e2e4'
  tertiary-fixed-dim: '#c8c6c8'
  on-tertiary-fixed: '#1b1b1d'
  on-tertiary-fixed-variant: '#474649'
  background: '#121317'
  on-background: '#e3e2e7'
  surface-variant: '#343539'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  title-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: 0em
  body-lg:
    fontFamily: Inter
    fontSize: 17px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  label-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: 0.02em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  container-padding: 24px
  stack-gap: 16px
  element-gap: 8px
  section-margin: 48px
  max-width-desktop: 1200px
---

## Brand & Style

The design system is rooted in **Minimalism** and **Modern Corporate** aesthetics, heavily inspired by Apple’s Human Interface Guidelines (HIG). The brand personality is professional, focused, and high-end, aiming to reduce the anxiety associated with technical interviews through a calm, monochromatic interface. 

The visual narrative centers on "The Void"—a deep, near-black environment where content is surfaced on elevated, subtly defined plates. The emotional response should be one of quiet confidence and precision. By removing unnecessary color and decoration, the user’s focus is directed entirely toward the simulation and their performance metrics.

## Colors

The palette is strictly monochromatic with a singular functional accent. 

- **Background**: The base layer uses `#0A0A0A` to create an infinite depth.
- **Surface**: UI elements like cards and containers use `#1C1C1E` to distinguish themselves from the background without harsh contrast.
- **Accents**: The color `#30D158` (iOS Green) is reserved exclusively for "Live" indicators, recording states, or success confirmations.
- **Typography**: High contrast is maintained using `#F5F5F7` for primary readability, while `#8E8E93` provides a hierarchy for secondary metadata and captions.
- **Borders**: Hairline strokes in `#2C2C2E` and `#3A3A3C` are used instead of shadows to define structure, mimicking the precision of hardware design.

## Typography

This design system utilizes **Inter** as a highly legible, systematic substitute for SF Pro, ensuring a premium "OS-level" feel. 

- **Hierarchy**: Use `display-lg` sparingly for welcome screens or score summaries. 
- **Body Text**: `body-lg` (17px) is the standard for interview questions to ensure maximum readability during high-pressure scenarios.
- **Clarity**: All labels use a medium weight to maintain legibility against dark surfaces. Tracking (letter spacing) is tightened for headlines and slightly loosened for small labels to optimize optical balance.

## Layout & Spacing

The layout follows a **Fixed Grid** philosophy for desktop to maintain a centered, cinematic focus, transitioning to a **Fluid Grid** for mobile.

- **Desktop**: A 12-column grid with a 1200px max-width. Content is typically centered in a 6-column or 8-column span to avoid eye strain.
- **Rhythm**: All spacing is derived from a 4px/8px baseline. Use `stack-gap` (16px) for related elements within a card and `container-padding` (24px) for internal padding of main surfaces.
- **Mobile**: Margins reduce to 16px. Typography scales down slightly while touch targets remain at a minimum of 44px.

## Elevation & Depth

Depth is conveyed through **Tonal Layering** and **Low-Contrast Outlines** rather than traditional shadows. 

- **Level 0 (Background)**: `#0A0A0A`. Used for the global canvas.
- **Level 1 (Cards)**: `#1C1C1E`. Surfaces sit directly on the background. They are defined by a 1px solid border of `#2C2C2E`.
- **Level 2 (Modals/Overlays)**: `#2C2C2E`. Elevated elements use a slightly lighter fill or a subtle `backdrop-filter: blur(20px)` if they are semi-transparent, mimicking the frosted glass effect of macOS and iOS.
- **Focus States**: Highlighting an active input or a selected card is achieved by changing the border color to `#3A3A3C` or adding a subtle white inner-glow (10% opacity).

## Shapes

The shape language is sophisticated and friendly, utilizing large corner radii to soften the high-contrast dark environment.

- **Main Containers**: Large cards (e.g., the interview simulator window) must use a `24px` radius.
- **Interactive Elements**: Buttons and input fields use a `12px` radius. 
- **Messaging**: Chat bubbles use a `14px` radius. For consecutive messages from the same sender, the "tail" corner should remain rounded while the other three maintain the standard radius to indicate flow.

## Components

- **Buttons**:
    - *Primary*: White (`#F5F5F7`) fill with black text. No border.
    - *Secondary*: Surface (`#1C1C1E`) fill with `#2C2C2E` border and white text.
- **Message Bubbles**: 
    - *AI/Interviewer*: Surface (`#1C1C1E`) with a left-aligned layout.
    - *User*: Deep Gray (`#2C2C2E`) with right-aligned layout.
- **Input Fields**: Ghost-style. No background fill, defined by a bottom border or a subtle `#2C2C2E` outline. Focus state changes the border to `#F5F5F7`.
- **Status Indicator**: A `10px` circle with a `#30D158` fill and a pulsing glow effect to represent "Live" audio or "AI Thinking" states.
- **Cards**: Use the `card_radius` (24px) and `#1C1C1E` fill. Borders are mandatory to ensure separation between similar tonal layers.
- **Lists**: Clean, divider-based rows using `#2C2C2E` as 0.5px hairlines. No icons unless they serve a functional purpose (e.g., file attachments).
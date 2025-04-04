# Streamlit Theming

This directory contains the theme configuration for the Personalized Drug AI application.

## Current Theme

The current theme uses a light base with soft pink background colors:

```toml
[theme]
base="light"
backgroundColor="#e4d5d5"
primaryColor="#d33682"
secondaryBackgroundColor="#f8e7e7"
textColor="#262730"
font="sans serif"
```

## Customizing the Theme

To modify the theme, edit the `config.toml` file with your preferred colors. Available properties include:

- `base`: Either "light" or "dark"
- `primaryColor`: Used for interactive elements
- `backgroundColor`: Main background color
- `secondaryBackgroundColor`: Background for sidebar and cards
- `textColor`: Color for body text
- `font`: Font family for text ("sans serif", "serif", or "monospace")

## Examples

### Dark Mode Theme
```toml
[theme]
base="dark"
backgroundColor="#1E1E1E"
primaryColor="#FF4B4B"
secondaryBackgroundColor="#262730"
textColor="#FAFAFA"
font="sans serif"
```

### Blue Clinical Theme
```toml
[theme]
base="light"
backgroundColor="#EBF2F8"
primaryColor="#1E88E5"
secondaryBackgroundColor="#FFFFFF"
textColor="#252525"
font="sans serif"
```

For more information, see the [Streamlit Theming Documentation](https://docs.streamlit.io/library/advanced-features/theming). 
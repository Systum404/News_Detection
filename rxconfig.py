import reflex as rx

config = rx.Config(
    app_name="News_Detection",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)
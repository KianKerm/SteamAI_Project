"""Compute and display compact library insights without model calls."""
import pandas as pd
import streamlit as st


# --- Calculate playtime summaries and genre membership ---
def build_insights(library):
    hours = pd.to_numeric(library['playtime_hours'], errors='coerce')
    hours = hours.where(hours.ge(0) & hours.lt(float('inf')))
    unplayed = hours.eq(0)
    brief = hours.gt(0) & hours.lt(2)

    # A multi-genre game contributes its full hours to each listed genre.
    games = library[['steam_appid', 'genres']].copy()
    games['Hours'] = hours
    games['Unplayed'] = unplayed
    games['Genre'] = games['genres'].fillna('').str.split(',')
    genres = games.explode('Genre')
    genres['Genre'] = genres['Genre'].str.strip()
    genres = genres.loc[genres['Genre'].ne('')].drop_duplicates(['steam_appid', 'Genre'])
    genre_hours = genres.groupby('Genre')['Hours'].sum(min_count=1).dropna()
    unplayed_genres = genres.loc[genres['Unplayed']].groupby('Genre').size()
    breakdown = pd.DataFrame({
        'Playtime': ['Unplayed', 'Under 2 hours', '2–10 hours', '10+ hours', 'Unknown'],
        'Games': [int(unplayed.sum()), int(brief.sum()),
                  int((hours.ge(2) & hours.lt(10)).sum()), int(hours.ge(10).sum()),
                  int(hours.isna().sum())],
    })
    total = hours.sum(min_count=1)
    return {
        'games': len(library), 'hours': None if pd.isna(total) else float(total),
        'unplayed': int(unplayed.sum()), 'brief': int(brief.sum()),
        'unknown': int(hours.isna().sum()),
        'missing_genres': int(library['genres'].fillna('').str.strip().eq('').sum()),
        'genre_hours': genre_hours.sort_values(ascending=False).head(8).rename('Hours').reset_index(),
        'unplayed_genres': unplayed_genres.sort_values(ascending=False).head(8).rename('Games').reset_index(),
        'breakdown': breakdown,
    }


# --- Display summary cards and one selected chart ---
def show_insights(library):
    data = build_insights(library)
    columns = st.columns(4)
    columns[0].metric('Games owned', f"{data['games']:,}")
    columns[1].metric('Recorded hours', 'Unknown' if data['hours'] is None else f"{data['hours']:,.1f}")
    columns[2].metric('Unplayed', f"{data['unplayed']:,}")
    columns[3].metric('Briefly played', f"{data['brief']:,}", help='More than zero, but less than two hours recorded.')
    view = st.radio('Explore your library',
                    ['Hours by genre', 'Unplayed genres', 'Playtime breakdown'], horizontal=True)
    if view == 'Hours by genre':
        chart, x, y = data['genre_hours'], 'Genre', 'Hours'
        st.caption('Top eight genres by recorded hours. Multi-genre games count fully toward each genre, so genre totals overlap.')
    elif view == 'Unplayed genres':
        chart, x, y = data['unplayed_genres'], 'Genre', 'Games'
        st.caption('Top eight genres among games with zero recorded hours. A game can appear in several genres.')
    else:
        chart, x, y = data['breakdown'], 'Playtime', 'Games'
        st.caption('Exclusive groups: zero; over zero to under 2; 2 to under 10; 10 or more hours; unknown.')
    if chart.empty:
        st.info('No matching data is available for this view.')
    else:
        st.bar_chart(chart, x=x, y=y, color=x, horizontal=True, sort=False)
    if data['unknown'] or data['missing_genres']:
        st.caption(f"Unknown playtime: {data['unknown']} games. Missing genre data: {data['missing_genres']} games. Missing values are excluded from the corresponding totals.")

#importing necessary libraries
import streamlit as st
import streamlit.components.v1 as stc
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#function for loading in data
def load_data(data):
    model_df=pd.read_csv(data)
    return model_df

model_df=load_data('final_data.csv')

#Function to vectorize + cosine similarity

def vectorize_cosine(data):
    Tvid=TfidfVectorizer(stop_words='english',analyzer='word')
    matrix=Tvid.fit_transform(data['tags'])
    #cosine
    similarity=cosine_similarity(matrix)
    return similarity

#function to recommend movie
def recommend(movie_title,similarity,model_df,num_of_rec):
    movie_indices=pd.Series(model_df.index,model_df['title']).drop_duplicates()
    #index of movie
    index=movie_indices[str(movie_title).title()]
    similarity_scores=list(enumerate(similarity[index]))
    sorted_sim_scores=sorted(similarity_scores,key=lambda x:x[1],reverse=True)
    rec_titles_indices=[i[0] for i in sorted_sim_scores[1:]]
    rec_sim=[i[1]for i in sorted_sim_scores[1:]]
    result_df=model_df.drop(columns='tags').iloc[rec_titles_indices]
    result_df['similarity']=rec_sim
    return result_df.head(num_of_rec)

#css templates for background and results
page_bg_img = '''
<style>
.stApp {
  background-image: url('https://www.bing.com/th?id=OIP.-_EYLAniPoYfSnSZHlgJXQHaHa&w=250&h=250&c=8&rs=1&qlt=90&o=6&dpr=1.25&pid=3.1&rm=2',%s');
  background-size: contain;
}
</style>
'''

RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 80px;
box-shadow:0 0 20px 5px #c000000; background-color:#db7093;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:white;"><span style="color:black;">😃Type::</span>{}</p>
<p style="color:white;"><span style="color:black;">📋Description:</span>{}</p>
<p style="color:white;"><span style="color:black;">🎥Genres:</span>{}</p>
<p style="color:white;"><span style="color:black;">📅Release Year:</span>{}</p>
<p style="color:white;"><span style="color:black;">💯Imdb Rating:</span>{}</p>
</div>
"""

#function to suggest other movies if query not in dataset
@st.cache
def search_term_if_not_found(term,model_df=model_df):
    term=term.lower()
    result_df = model_df[model_df['title'].astype('str').apply(lambda x:x.lower()).str.contains(term)]
    if result_df.shape[0]==0:
        #st.info('Not in database')
        result_df=model_df.head(10)
    return result_df

#main function
def main():
    st.markdown(page_bg_img, unsafe_allow_html=True)

    #title
    st.title('Netflix Recommender System Demo')
    menu=['Home','Recommend',"Visualizations",'About']

    choice=st.sidebar.selectbox("Menu",menu)
   
    model_df=load_data('final_data.csv')

    if choice=='Home':

        st.subheader('Home')
        st.write("This is a demo of a content based filtering recommender system with the documentation on github, explore!")
        st.write(f"The data used to build this system contains exactly {model_df.shape[0]} unique movies and {model_df.shape[1]} features")
        st.markdown("""     The first 10 movies of the data are displayed below
        """)
        st.dataframe(model_df.drop(columns='tags').set_index('id').head(10))
        st.write("The recommender system in the 'Recommend' Menu was built With the use of Tfidf Vectorizer and Cosine similarity")

    elif choice=='Recommend':
        st.subheader('Recommend Movies')
        num_of_rec = st.sidebar.number_input("Number",1,30)
        similarity=vectorize_cosine(model_df)
        search_term=st.text_input('Search Movie')
        if st.button('Recommend'):
            if search_term!= None:
                try:
                    result=recommend(search_term,similarity,model_df,num_of_rec=num_of_rec)
                    with st.expander("Results as JSON"):
                        results_json = result.to_dict('index')
                        st.write(results_json)
						
                    for row in result.iterrows():
                        rec_title = row[1][1]
                        rec_type = str(row[1][2]).title()
                        rec_desc = row[1][3]
                        rec_gen  = str(row[1][7]).title()
                        rec_year = row[1][4]
                        rec_imdb = row[1][-3]

                        stc.html(RESULT_TEMP.format(rec_title,rec_type,rec_desc,rec_gen,rec_year,rec_imdb),height=350)		
                    
                except :
                    result='Not in Database'
                    st.warning(result)
                    st.info("Suggested Movies include")
                    result_df = search_term_if_not_found(search_term,model_df)
                    for row in result_df.iterrows():
                        rec_title = row[1][1]
                        rec_type = row[1][2]
                        rec_desc = row[1][3]
                        rec_gen  = row[1][7]
                        rec_year = row[1][4]
                        rec_imdb = row[1][-3]

                        stc.html(RESULT_TEMP.format(rec_title,rec_type,rec_desc,rec_gen,rec_year,rec_imdb),height=350)	
                    st.dataframe(result_df)
    elif choice == "Visualizations":
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #data visualizations
        st.subheader("Exploring Correlations")

        fig=plt.figure(figsize=[12,6])
        sns.scatterplot(data=model_df,hue='type',x='imdb_score',y='tmdb_score')
        plt.title(f"Correlation: {model_df['imdb_score'].corr(model_df['tmdb_score'])}")
        st.pyplot(fig)
        
        fig=plt.figure(figsize=[12,6])
        sns.scatterplot(data=model_df,hue='type',x='seasons',y='runtime')
        plt.title(f"Correlation: {model_df['seasons'].corr(model_df['runtime'])}")
        st.pyplot(fig)

        st.subheader("Distribution of Numeric columns")

        for col in ['imdb_score', 'tmdb_score']:
            fig=plt.figure(figsize=[12,6])
            sns.histplot(model_df[col])
            plt.title(f"{col} distribution")
            
            st.pyplot(fig)

        for col in ['tmdb_popularity', 'imdb_votes']:
            fig=plt.figure(figsize=[12,6])
            sns.histplot(model_df[col],bins=30)
            plt.title(f"{col} distribution")
            
            st.pyplot(fig)

        st.subheader('Distribution of categorical variables')

        for col in ['age_certification','seasons']:
            fig=plt.figure(figsize=[12,6])
            palette=sns.color_palette(n_colors=1)
            sns.countplot(model_df[col],palette=palette)
            plt.title(f"{col} distribution")
            
            st.pyplot(fig)

        genre=load_data('gen.csv')
        countries=load_data('pdt.csv').head(10)

        fig=plt.figure(figsize=[12,6])
        sns.barplot(data=genre,x='counts',y='genre',palette=sns.color_palette(n_colors=1))
        plt.title('Genres count in dataset')
        st.pyplot(fig)

        fig=plt.figure(figsize=[12,6])
        sns.barplot(data=countries,x='counts',y='production_countries',palette=sns.color_palette(n_colors=1))
        plt.title('Production countries count in dataset')
        st.pyplot(fig)

    else:
        st.subheader('About')
        st.text("Made with streamlit by Theresa Sunday(Resa200) on github")




if __name__== '__main__':
     main()

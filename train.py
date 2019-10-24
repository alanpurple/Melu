import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import metrics,Model,layers,Sequential,losses,optimizers,utils
from connect_db import Session,engine
from data_models import Movie,User,Rating
import json
from melu_model import MeluGlobal,MeluLocal
from sqlalchemy import func
from math import floor

# get all rating
# divide movies before 1997 and after 1998 ( approximately 8:2 )

# divide user into new and existing group
# remove rating for existing items rated by new users
# remove rating for new items rated by existing users

MOVIE_MIN_YEAR=1919
MOVIE_MAX_YEAR=2000
MAX_USER_ID=6040
scenario_len=40
query_len=10
validatioin_len=4
alpha=0.01
beta=0.01

def main():
    session=Session()
    # query with condition? alternative
    all_users=session.query(User).all()
    all_movies=session.query(Movie).all()

    user_rating_counts=session.query(Rating.user_id,func.count(Rating.user_id)).group_by(Rating.user_id).all()

    # user with more than 40 ratings
    user_filtered=filter(lambda x: x[1]>60,user_rating_counts)
    actual_users_index=[elem[0] for elem in user_filtered]

    actor_dict,director_dict,rated_dict,genre_dict=get_movie_dict('movie_dict.json')
    #author_dict,publisher_dict=get_book_dict('book_dict.json')
    
    with open('movie_user_zipcodes.json','r') as f:
        zipcodes=json.load(f)
    zipcode_dict=dict(zip(zipcodes,range(len(zipcodes))))

    all_users_id=[elem.id for elem in all_users]
    all_users_data=[{'gender':elem.gender,'occupation':elem.occupation,'age':elem.age,'zipcode':elem.zipcode} for elem in all_users]

    all_users_df=pd.DataFrame(all_users_data,index=all_users_id)

    # occupation doesn't need hashing
    occu_dict_size=all_users_df.occupation.max()+1

    all_users_df.gender=(all_users_df.gender=='M').astype(int)
    all_users_df.zipcode=all_users_df.zipcode.apply(lambda x: zipcode_dict[x])

    user_ages=sorted(all_users_df.age.unique())
    # age may be quantifiable, but every person in their age periods has their own culture and style 
    age_dict=dict(zip(user_ages,range(len(user_ages))))

    all_users_df.age=all_users_df.age.apply(lambda x:age_dict[x])

    all_movies_id=[elem.id for elem in all_movies]
    all_movies_data=[{
        'year':elem.year,'actor':elem.actor,'title':elem.title,'rated':elem.rated,
        'director':elem.director,'genre':elem.genre
        } for elem in all_movies]

    all_movies_df=pd.DataFrame(all_movies_data,index=all_movies_id)

    all_movies_df.actor=all_movies_df.actor.apply(lambda x: actor_dict[x])
    all_movies_df.director=all_movies_df.director.apply(lambda x: director_dict[x])
    all_movies_df.rated=all_movies_df.rated.apply(lambda x: rated_dict[x])
    all_movies_df.genre=all_movies_df.genre.apply(lambda x: genre_dict[x])
    all_movies_df.year=all_movies_df.year - MOVIE_MIN_YEAR

    existing_movies_df=all_movies_df[all_movies_df.year<1998-MOVIE_MIN_YEAR]
    new_movies_df=all_movies_df[all_movies_df.year>1997-MOVIE_MIN_YEAR]

    #user_mask=np.random.rand(len(all_users_df)) < 0.8
    #user_existing=all_users_df[user_mask]
    #user_new=all_users_df[~user_mask]
    user_existing=all_users_df[all_users_df.index.isin(actual_users_index)]
    user_new=all_users_df[~all_users_df.index.isin(actual_users_index)]


    rating_existing=session.query(Rating).join(User).filter(User.id.in_(user_existing.index)).join(Movie).filter(Movie.year<1998).all()
    #rating_exist_new=session.query(Rating).join(User).filter(User.id.in_(user_existing.index)).join(Movie).filter(Movie.year>1997).all()
    #rating_new_exist=session.query(Rating).join(User).filter(User.id.in_(user_new.index)).join(Movie).filter(Movie.year<1998).all()
    #rating_new_new=session.query(Rating).join(User).filter(User.id.in_(user_new.index)).join(Movie).filter(Movie.year>1997).all()

    '''
    train_genders=[1 if elem.user.genre=='M' else 0 for elem in rating_existing]
    train_occupations=[elem.user.occupation for elem in rating_existing]
    train_ages=[elem.user.age for elem in rating_existing]
    train_zipcodes=[all_users_df.loc[elem.user_id].zipcode for elem in rating_existing]
    train_actors=[all_movies_df.loc[elem.movie_id].actor for elem in rating_existing]
    train_directors=[all_movies_df.loc[elem.movie_id].director for elem in rating_existing]
    train_genres=[all_movies_df.loc[elem.movie_id].genre for elem in rating_existing]
    train_rateds=[all_movies_df.loc[elem.movie_id].rated for elem in rating_existing]

    train_labels=[(elem.rate-1)*0.25 for elem in rating_existing]
    '''

    rating_existing_group=[[] for _ in range(MAX_USER_ID+1)]
    for rating in rating_existing:
        # 40 ratings per user, + 10 queries
        if len(rating_existing_group[rating.user_id])<scenario_len+query_len:
            rating_existing_group[rating.user_id].append(rating)

    actual_users_index2=[idx for idx,elem in enumerate(rating_existing_group) if len(elem)>scenario_len+query_len-1]
    

    dict_sizes={'zipcode':len(zipcode_dict),'actor':len(actor_dict),
                'authdir':len(director_dict),'rated':len(rated_dict),
                'year':MOVIE_MAX_YEAR-MOVIE_MIN_YEAR+1,'occu':occu_dict_size,
                'age':len(age_dict),'genre':len(genre_dict)}
    emb_sizes={'zipcode':100,'actor':50,'authdir':50,'rated':5,'year':15,'occu':4,'age':2,'genre':15}

    global_model=MeluGlobal(dict_sizes,emb_sizes,1)
    emb_input_size=sum([v for k,v in emb_sizes.items()])
    local_model=MeluLocal(emb_input_size,[64,32,16,4])

    print(global_model.summary())
    print(local_model.summary())
    utils.plot_model(global_model,'global.png',True,expand_nested=True)
    utils.plot_model(local_model,'local.png',True,expand_nested=True)

    USER_BATCH_SIZE=128

    # task batch size should divide scenario length
    TASK_BATCH_SIZE=20

    total_batch=floor(len(actual_users_index2)/USER_BATCH_SIZE)
    #remaining_users=len(actual_users_index2)%USER_BATCH_SIZE

    local_loss_fn=losses.MeanAbsoluteError()
    local_optimizer=optimizers.Adam(alpha)
    global_optimizer=optimizers.Adam(beta)
    #global_loss_fn=losses.MeanAbsoluteError()

    #local_model.compile(local_optimizer,local_loss_fn,[metrics.MeanAbsoluteError()])
    #global_model.compile(global_optimizer,global_loss_fn,[metrics.MeanAbsoluteError()])

    #local_model.save_weights('theta2.h5')
    local_model_weights=local_model.get_weights()

    # prepare training metric
    #val_metric=metrics.MeanAbsoluteError()
    for epoch in range(30):
        print('start epoch {}'.format(epoch))
        # previous validation loss to decide early stopping
        # prev_val_loss - epoch-1 loss
        # prev2_val_loss - epoch-2 loss
        # prev3_val_loss - epoch-3 loss
        if epoch>19:
            prev3_train_loss=prev2_train_loss
            prev2_train_loss=prev_train_loss
            prev_train_loss=total_train_loss
        elif epoch==19:
            prev2_train_loss=prev_train_loss
            prev_train_loss=total_train_loss
        elif epoch==18:
            prev_train_loss=total_train_loss
        total_train_loss=0
        for i in range(total_batch):
            print('user batch # {}'.format(i))
            users=[rating_existing_group[elem] for elem in actual_users_index2[i*USER_BATCH_SIZE:(i+1)*USER_BATCH_SIZE]]

            theta2_user_weights=[]

            

            # calculate local weights per user
            for j,user in enumerate(users):
                #local_model.load_weights('theta2.h5')
                local_model.set_weights(local_model_weights)
                # [authdir,year,age,actor,rated,genre,occu,zipcode]
                user_data=[
                    [existing_movies_df.loc[elem.movie_id].director,
                    existing_movies_df.loc[elem.movie_id].year,
                    all_users_df.loc[elem.user_id].age,
                    existing_movies_df.loc[elem.movie_id].actor,
                    existing_movies_df.loc[elem.movie_id].rated,
                    existing_movies_df.loc[elem.movie_id].genre,
                    all_users_df.loc[elem.user_id].occupation,
                    all_users_df.loc[elem.user_id].zipcode
                    ] for elem in user[:scenario_len]
                ]
                label_data=[elem.rate for elem in user[:scenario_len]]
                train_dataset=tf.data.Dataset.from_tensor_slices((user_data,label_data)).batch(TASK_BATCH_SIZE,True)
                for (user_batch,label_batch) in train_dataset:
                    batch_emb_out=global_model(user_batch)
                    with tf.GradientTape() as tape:
                        logits=local_model(batch_emb_out)
                        local_loss=local_loss_fn(label_batch,logits)
                    local_grads=tape.gradient(local_loss,local_model.trainable_weights)
                    local_optimizer.apply_gradients(zip(local_grads,local_model.trainable_weights))
                #local_model.save_weights('theta2_{}.h5'.format(j))
                theta2_user_weights.append(local_model.get_weights())
            # calculate gradients for each uesr
            theta1_grads=[]
            theta1_losses=0
            for j,user in enumerate(users):
                #local_model.load_weights('theta2_{}.h5'.format(j))
                local_model.set_weights(theta2_user_weights[j])
                user_query=[
                    [existing_movies_df.loc[elem.movie_id].director,
                    existing_movies_df.loc[elem.movie_id].year,
                    all_users_df.loc[elem.user_id].age,
                    existing_movies_df.loc[elem.movie_id].actor,
                    existing_movies_df.loc[elem.movie_id].rated,
                    existing_movies_df.loc[elem.movie_id].genre,
                    all_users_df.loc[elem.user_id].occupation,
                    all_users_df.loc[elem.user_id].zipcode
                    ] for elem in user[scenario_len:]
                ]
                label_data=[elem.rate for elem in user[scenario_len:]]
                train_dataset=tf.data.Dataset.from_tensor_slices((user_query,label_data)).batch(query_len)
                (query_batch,label_batch)=next(iter(train_dataset))
                with tf.GradientTape() as tape:
                    emb_out=global_model(query_batch)
                    logits=local_model(emb_out)
                    local_loss=local_loss_fn(label_batch,logits)
                    theta1_losses+=local_loss
                    # there will be USER_BATCH_SIZE * scenario_len/TASK_BATCH_SIZE gradients
                grad=tape.gradient(local_loss,global_model.trainable_weights)
                global_optimizer.apply_gradients(zip(grad,global_model.trainable_weights))
                theta1_grads.append(grad)
            # apply every gradients to embedding layer weights
            final_theta1_grad=[]
            theata2_losses=0
            for k in range(len(theta1_grads[0])):
                data=[elem[k] for elem in theta1_grads]
                final_data=tf.add_n(data)/USER_BATCH_SIZE
                final_theta1_grad.append(final_data)
            global_optimizer.apply_gradients(zip(final_theta1_grad,global_model.trainable_weights))

            # calculate each local gradients per user for updated global theta1
            theta2_grads=[]
            for j,user in enumerate(users):
                #local_model.load_weights('theta2_{}.h5'.format(j))
                # below line is wrong(maybe)
                #local_model.set_weights(theta2_user_weights[j])
                local_model.set_weights(local_model_weights)
                user_query=[
                    [existing_movies_df.loc[elem.movie_id].director,
                    existing_movies_df.loc[elem.movie_id].year,
                    all_users_df.loc[elem.user_id].age,
                    existing_movies_df.loc[elem.movie_id].actor,
                    existing_movies_df.loc[elem.movie_id].rated,
                    existing_movies_df.loc[elem.movie_id].genre,
                    all_users_df.loc[elem.user_id].occupation,
                    all_users_df.loc[elem.user_id].zipcode
                    ] for elem in user[scenario_len:]
                ]
                label_data=[elem.rate for elem in user[scenario_len:]]
                train_dataset=tf.data.Dataset.from_tensor_slices((user_query,label_data)).batch(query_len)
                (query_batch,label_batch)=next(iter(train_dataset))
                emb_out=global_model(query_batch)
                with tf.GradientTape() as tape:
                    logits=local_model(emb_out)
                    local_loss=local_loss_fn(label_batch,logits)
                    theata2_losses+=local_loss
                theta2_grads.append(tape.gradient(local_loss,local_model.trainable_weights))
            # update global dense layer weights
            #local_model.load_weights('theta2.h5')
            local_model.set_weights(local_model_weights)
            final_theta2_grad=[]
            for k in range(len(theta2_grads[0])):
                data=[elem[k] for elem in theta2_grads]
                final_data=tf.add_n(data)/USER_BATCH_SIZE
                final_theta2_grad.append(final_data)
            global_optimizer.apply_gradients(zip(final_theta2_grad,local_model.trainable_weights))
            #local_model.save_weights('theta2.h5')
            local_model_weights=local_model.get_weights()

            # To Do: evaluate validation
            # use MAE ( paper's choice )
            '''
            batch_val_loss=0
            for j,user in enumerate(users):
                validation_batch=user[scenario_len:scenario_len+validatioin_len]   # this is actually all of it
                batch_input=[
                    [existing_movies_df.loc[elem.movie_id].director,
                    existing_movies_df.loc[elem.movie_id].year,
                    all_users_df.loc[elem.user_id].age,
                    existing_movies_df.loc[elem.movie_id].actor,
                    existing_movies_df.loc[elem.movie_id].rated,
                    existing_movies_df.loc[elem.movie_id].genre,
                    all_users_df.loc[elem.user_id].occupation,
                    all_users_df.loc[elem.user_id].zipcode
                    ] for elem in validation_batch
                ]
                batch_labels=[elem.rate for elem in validation_batch]

                # only one batch, so need to be in one-item list
                val_embedded=global_model.predict_on_batch([batch_input])
                val_logits=local_model.predict_on_batch(val_embedded)
                val_metric(batch_labels,val_logits)
                batch_val_loss=batch_val_loss+val_metric.result()
            

            print('validation loss: %s' % (float(batch_val_loss),))
            total_train_loss+=batch_val_loss
            # To do: end train if validation loss increases of not be reduced enogh - Early stopping
            '''
            #measure total training loss
            print('batch #{} theta1 loss:{}'.format(i,theta1_losses))
            print('batch #{} theta2 loss:{}'.format(i,theata2_losses))
            total_train_loss+=theta1_losses+theata2_losses

        if epoch%5==0:
            local_model.save('models/local_model_{}.h5'.format(epoch))
            global_model.save('models/global_model_{}.h5'.format(epoch))
        if epoch>19:
            pass
            min_prev_loss=min([prev_train_loss,prev2_train_loss,prev3_train_loss])
            print('previous train loss: ',min_prev_loss)
            print('current train loss at epoch {}: '.format(epoch), total_train_loss)
            if total_train_loss>min_prev_loss:
                print('total train loss increases, end training')
                break

    local_model.save('models/local_model_{}_final.h5'.format(epoch))
    global_model.save('models/global_model_{}_final.h5'.format(epoch))
                


def get_movie_dict(movie_dict_file):
    with open(movie_dict_file,'r') as f:
        movie_dict=json.load(f)
    actor_dict=dict(zip(movie_dict['actors'],range(len(movie_dict['actors']))))
    director_dict=dict(zip(movie_dict['directors'],range(len(movie_dict['directors']))))
    rated_dict=dict(zip(movie_dict['rateds'],range(len(movie_dict['rateds']))))
    genre_dict=dict(zip(movie_dict['genres'],range(len(movie_dict['genres']))))
    return actor_dict,director_dict,rated_dict,genre_dict

def get_book_dict(book_dict_file):
    with open(book_dict_file,'r') as f:
        book_dict=json.load(f)
    author_dict=dict(zip(book_dict['authors'],range(len(book_dict['authors']))))
    publisher_dict=dict(zip(book_dict['publishers'],range(len(book_dict['publishers']))))
    return author_dict,publisher_dict

def ndcg(label,pred):
    pass

def dcg():
    pass

if __name__=='__main__':
    main()
import twitter

# The convenient way to get access to your own data
def oauth_login():
  CONSUMER_KEY = "[your own key]"
  CONSUMER_SECRET = "[your own secret]"

  OAUTH_TOKEN = "[your own token]"
  OAUTH_TOKEN_SECRET = "[your own secret]"

  auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
  twitter_api = twitter.Twitter(auth = auth)
  
  return twitter_api

twitter_api = oauth_login()

print twitter_api

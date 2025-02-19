import streamlit_authenticator as stauth

# Define your users' credentials
credentials = {
    'usernames': {
        'oracle': {
            'email': 'oracle9@gmail.com',
            'name': 'Oracle',
            'password': 'oracle'
        },
        'yk': {
            'email': 'yk9@gmail.com',
            'name': 'YK',
            'password': 'lacoona'
        }
    }
}

# Hash the passwords
hashed_credentials = stauth.Hasher.hash_passwords(credentials)

# Print the hashed credentials
for username, user_info in hashed_credentials['usernames'].items():
    print(f"Username: {username}")
    print(f"Email: {user_info['email']}")
    print(f"Name: {user_info['name']}")
    print(f"Hashed Password: {user_info['password']}")
    print("-" * 40)


import os
key = os.urandom(24).hex()
print(key)

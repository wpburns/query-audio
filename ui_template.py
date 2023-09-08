css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.response {
    background-color: #475063
}

.chat-message .message {
  width: 95%;
  padding: 0 1.5rem;
  color: #fff;
  text-align: justify;
  text-justify: inter-word;
}
'''

response_template = '''
<div class="chat-message response">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">   
    <div class="message">{{MSG}}</div>
</div>
'''
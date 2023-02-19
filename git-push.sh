
git_push() {

git add .
git commit -m "$1"
git push -u origin master
}


echo "Message to Push?"
read message
git_push "$message"




# changing the 8th lines  [sed -i 'Ns/.*/replacement-line/' file.txt]
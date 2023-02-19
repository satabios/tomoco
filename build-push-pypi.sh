
package_build_push() {

sed -i  "8s/.*/version = \"$1\"/" pyproject.toml  
python3 setup.py sdist bdist_wheel 
twine upload "dist/tomoco-""$1""*"


git add .
git commit -m "Pushed Version $1"
git push -u origin master
}


echo "Version to Build?"
read version
package_build_push "$version"




# changing the 8th lines  [sed -i 'Ns/.*/replacement-line/' file.txt]
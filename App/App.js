import 'react-native-gesture-handler';
import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import Index from './src/index';
import Result from './src/result';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="FaceApp" component={Index} />
        <Stack.Screen name="Kết quả" component={Result} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

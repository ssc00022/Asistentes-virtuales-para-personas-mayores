import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

import MainScreen from "./MainScreen";
import FormScreen from "./FormScreen";

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    // Contenedor principal de navegación
    <NavigationContainer>
      <Stack.Navigator>
        {/* Pantalla de conversación con el asistente */}
        <Stack.Screen name="Chat" component={MainScreen} />
        {/* Pantalla donde el usuario introduce su información personal */}
        <Stack.Screen name="Perfil de usuario" component={FormScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export interface User {
  id: string;
  username: string;
  email: string;
  isActive: boolean;
}

export class UserService {
  private apiBaseUrl: string;

  constructor(apiBaseUrl: string = "https://api.example.com") {
    this.apiBaseUrl = apiBaseUrl;
  }

  async getUserById(userId: string): Promise<User> {
    const response = await fetch(`${this.apiBaseUrl}/users/${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch user with id: ${userId}`);
    }
    return response.json();
  }

  async deactivateUser(userId: string): Promise<boolean> {
    const response = await fetch(`${this.apiBaseUrl}/users/${userId}/deactivate`, {
      method: "POST",
    });
    return response.ok;
  }
}

export const formatUserDisplayName = (user: User): string => {
  return `${user.username} <${user.email}>`;
};
